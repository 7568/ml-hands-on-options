import osimport sysimport torchimport torch.nn as nn# Append the library path to PYTHONPATH, so library can be imported.# sys.path.append(os.path.dirname(os.getcwd()))from tqdm import tqdmsys.path.append(os.path.dirname("../../*"))sys.path.append(os.path.dirname("../*"))from torch.utils.data import DataLoaderimport numpy as npimport pandas as pdimport randomimport timefrom hedging_options.library import datasetimport transformer_net# from tqdm import tqdmimport loggingSEED = 1234random.seed(SEED)np.random.seed(SEED)torch.manual_seed(SEED)torch.cuda.manual_seed(SEED)torch.backends.cudnn.deterministic = Trueif not os.path.exists('log'):    os.makedirs('log')if not os.path.exists('pt'):    os.makedirs('pt')if not os.path.exists('pid'):    os.makedirs('pid')def train(model, optimizer, clip, batch_size, train_dataloader, training_dataset_length):    model.train()    all_loss = []    for ii, (datas, results) in tqdm(enumerate(train_dataloader), total=training_dataset_length / batch_size):    # for ii, (datas, results) in enumerate(train_dataloader):        datas = datas.float().to(DEVICE)        results = results.float().to(DEVICE)        output = model(datas, results)        if torch.isnan(output).any():            print(ii)        output_dim = output.shape[0]        R_1 = results.contiguous().view(output_dim, -1)        y = R_1[:, -2].view(output_dim, -1)        # print(y-y_hat)        # v_1 = output.contiguous().view(-1, output_dim)[:, -1]        # trg = results.contiguous().view(-1)        loss = criterion(output, y)        loss2 = criterion2(output, y)        # print(output,' ===== ',y)        # print(loss.item(),loss2.item())        optimizer.zero_grad()        if (loss.item() > loss2.item()) & (loss.item() < 7):            loss.backward()        else:            loss2.backward()        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)        optimizer.step()        # print('=============================')        # for name, parameters in model.named_parameters():        #     if name == 'encoder.layers.0.self_attention.fc_q.weight':        #         print('\n')        #         print(parameters.grad.data[0,:5],'       ',parameters.data[0,:5])        # print(name, ':', parameters.size(),' : ', parameters.grad)        # print('=============================')        # epoch_loss += loss.item()        all_loss.append(loss.item())        # print(loss.item())    return np.array(all_loss).mean()def evaluate(model, batch_size, val_dataloader, valid_data_length):    model.eval()    # epoch_loss = 0    all_loss = []    with torch.no_grad():        for ii, (datas, results) in tqdm(enumerate(val_dataloader), total=valid_data_length / batch_size):            # for ii, (datas, results) in enumerate(val_dataloader):            datas = datas.float().to(DEVICE)            results = results.float().to(DEVICE)            output = model(datas, results)            output_dim = output.shape[0]            R_1 = results.contiguous().view(output_dim, -1)            y = R_1[:, -2].view(output_dim, -1)            # print(y-y_hat)            # v_1 = output.contiguous().view(-1, output_dim)[:, -1]            # trg = results.contiguous().view(-1)            loss = criterion(output, y)            # epoch_loss += loss.item()            all_loss.append(loss.item())            # if math.isnan(epoch_loss):            #     print(epoch_loss)    return np.array(all_loss).mean()def get_mshe_in_test(model, batch_size, test_dataloader, test_dataset_length):    model.eval()    mshes_put = []    mshes_call = []    with torch.no_grad():        for ii, (datas, results) in tqdm(enumerate(test_dataloader), total=test_dataset_length / batch_size):            # for ii, (datas, results) in enumerate(test_dataloader):            datas = datas.float().to(DEVICE)            results = results.float().to(DEVICE)            output = model(datas, results)            output_dim = output.shape[0]            R_1 = results.contiguous().view(output_dim, -1)            y = R_1[:, -2].view(output_dim, -1)            # v_1 = output.contiguous().view(-1, output_dim)[:, -1]            # trg = results.contiguous().view(-1)            mshe = torch.pow((100 * (output - y).view(output_dim)) / R_1[:, -1], 2).detach().cpu().numpy()            # mshes = np.append(mshes, mshe)            R_1_cpu = R_1.detach().cpu().numpy()            mshe_put = mshe[R_1_cpu[:, 1] == 1]            mshe_call = mshe[R_1_cpu[:, 1] == 0]            if len(mshe_put) > 0:                mshes_put = np.append(mshes_put, mshe_put)            if len(mshe_call) > 0:                mshes_call = np.append(mshes_call, mshe_call)    return np.mean(mshes_put), np.mean(mshes_call)def epoch_time(start_time, end_time):    elapsed_time = end_time - start_time    elapsed_mins = int(elapsed_time / 60)    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))    return elapsed_mins, elapsed_secsdef load_data(batch_size):    train_data_index = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300_training_index.csv').to_numpy()    valid_data_index = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300_validation_index.csv').to_numpy()    test_data_index = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300_test_index.csv').to_numpy()    # train_data_index = train_data_index[10000:, :]    # valid_data_index = valid_data_index[:3000, :]    training_dataset = dataset.Dataset_transformer(train_data_index, f'{PREPARE_HOME_PATH}/parquet/training/')    valid_dataset = dataset.Dataset_transformer(valid_data_index, f'{PREPARE_HOME_PATH}/parquet/validation/')    test_dataset = dataset.Dataset_transformer(test_data_index, f'{PREPARE_HOME_PATH}/parquet/test/')    _train_dataloader = DataLoader(training_dataset, num_workers=NUM_WORKERS, batch_size=batch_size, shuffle=True)    _val_dataloader = DataLoader(valid_dataset, num_workers=NUM_WORKERS, batch_size=batch_size)    _test_dataloader = DataLoader(test_dataset, num_workers=NUM_WORKERS, batch_size=batch_size)    return _train_dataloader, _val_dataloader, _test_dataloader, len(training_dataset), len(valid_dataset), len(        test_dataset)N_EPOCHS = 10CLIP = 1# gpu_ids = '0,1,2'# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idsENC_LAYERS = 3DEVICE = 'cpu'INTPUT_DIM = 20HID_DIM = 56ENC_HEADS = 4ENC_DROPOUT = 0.1IS_TRAIN = True# 3,6,9,16# print(f'ENC_LAYERS : {ENC_LAYERS}')# python transformer-code-comments.py > 0.0005-log &PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/'# PREPARE_HOME_PATH = f'/home/zhanghu/liyu/data/'NUM_WORKERS = 0LEARNING_RATE = 0.0002BATCH_SIZE = 8criterion = nn.MSELoss()criterion2 = nn.L1Loss()if __name__ == '__main__':    logger = logging.getLogger()    logger.setLevel(level=logging.DEBUG)    if len(sys.argv) > 1:        ENC_LAYERS = int(sys.argv[1])        # print(f'CUDA_VISIBLE_DEVICES : {sys.argv[2]}')        # torch.cuda.set_device(int(sys.argv[2]))        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(sys.argv[2]))        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        sys.stderr = open('log-transformer_train_code.log', 'a')    TRAINING_DATASET_LENGTH = 0    VALID_DATASET_LENGTH = 0    TEST_DATASET_LENGTH = 0    train_dataloader, val_dataloader, test_dataloader, training_dataset_length, valid_dataset_length, test_dataset_length = load_data(BATCH_SIZE)    BEST_VALID_LOSS = float('inf')    handler = logging.FileHandler(f'log/train_{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}.log')    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')    handler.setFormatter(formatter)    logger.addHandler(handler)    logger.debug(f'{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}')    # Create data loaders.    # writer = SummaryWriter()    enc = transformer_net.Encoder(INTPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_DROPOUT, DEVICE)    model = transformer_net.Seq2Seq(enc, INTPUT_DIM, DEVICE).to(DEVICE)    logger.debug(f'The model has {transformer_net.count_parameters(model):,} trainable parameters')    model.apply(transformer_net.initialize_weights)    # model = torch.nn.DataParallel(model)    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)    # logger.debug(f'MSHE in test : {get_mshe_in_test(model, BATCH_SIZE)}')    if IS_TRAIN:        for epoch in range(N_EPOCHS):            start_time = time.time()            train_loss = train(model, optimizer,  CLIP, BATCH_SIZE, train_dataloader, training_dataset_length)            valid_loss = evaluate(model, BATCH_SIZE, val_dataloader, valid_dataset_length)            end_time = time.time()            epoch_mins, epoch_secs = epoch_time(start_time, end_time)            if valid_loss < BEST_VALID_LOSS:                BEST_VALID_LOSS = valid_loss                torch.save(model.state_dict(), f'pt/{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}-model.pt')            logger.debug(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')            logger.debug(f'Train Loss: {train_loss:.3f} ')            logger.debug(f'validate Loss: {valid_loss:.3f} ')            if epoch % 3 == 0:                logger.debug(                    f'MSHE in test : {get_mshe_in_test(model, BATCH_SIZE, test_dataloader, test_dataset_length)}')    model.load_state_dict(torch.load(f'pt/{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}-model.pt'))    logger.debug(f'MSHE in test : {get_mshe_in_test(model, BATCH_SIZE, test_dataloader, test_dataset_length)}')## def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):#     model.eval()##     if isinstance(sentence, str):#         nlp = spacy.load('de_core_news_sm')#         tokens = [token.text.lower() for token in nlp(sentence)]#     else:#         tokens = [token.lower() for token in sentence]##     tokens = [src_field.init_token] + tokens + [src_field.eos_token]##     src_indexes = [src_field.vocab.stoi[token] for token in tokens]##     src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)##     src_mask = model.make_src_mask(src_tensor)##     with torch.no_grad():#         enc_src = model.encoder(src_tensor, src_mask)##     trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]##     for i in range(max_len):##         trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)##         trg_mask = model.make_trg_mask(trg_tensor)##         with torch.no_grad():#             output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)##         pred_token = output.argmax(2)[:, -1].item()##         trg_indexes.append(pred_token)##         if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:#             break##     trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]##     return trg_tokens[1:], attention### def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):#     assert n_rows * n_cols == n_heads##     fig = plt.figure(figsize=(15, 25))##     for i in range(n_heads):#         ax = fig.add_subplot(n_rows, n_cols, i + 1)##         _attention = attention.squeeze(0)[i].cpu().detach().numpy()##         cax = ax.matshow(_attention, cmap='bone')##         ax.tick_params(labelsize=12)#         ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],#                            rotation=45)#         ax.set_yticklabels([''] + translation)##         ax.xaxis.set_major_locator(ticker.MultipleLocator(1))#         ax.yaxis.set_major_locator(ticker.MultipleLocator(1))##     plt.show()#     plt.close()### example_idx = 8## src = vars(train_data_map.examples[example_idx])['src']# trg = vars(train_data_map.examples[example_idx])['trg']## print(f'src = {src}')# print(f'trg = {trg}')## translation, attention = translate_sentence(src, SRC, TRG, model, DEVICE)## print(f'predicted trg = {translation}')## display_attention(src, translation, attention)## example_idx = 6## src = vars(valid_data.examples[example_idx])['src']# trg = vars(valid_data.examples[example_idx])['trg']## print(f'src = {src}')# print(f'trg = {trg}')# translation, attention = translate_sentence(src, SRC, TRG, model, DEVICE)## print(f'predicted trg = {translation}')## display_attention(src, translation, attention)## example_idx = 10## src = vars(test_data.examples[example_idx])['src']# trg = vars(test_data.examples[example_idx])['trg']## print(f'src = {src}')# print(f'trg = {trg}')## translation, attention = translate_sentence(src, SRC, TRG, model, DEVICE)## print(f'predicted trg = {translation}')## display_attention(src, translation, attention)### def calculate_bleu(data, src_field, trg_field, model, device, max_len=50):#     trgs = []#     pred_trgs = []##     for datum in data:#         src = vars(datum)['src']#         trg = vars(datum)['trg']##         pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)##         # cut off <eos> token#         pred_trg = pred_trg[:-1]##         pred_trgs.append(pred_trg)#         trgs.append([trg])##     return bleu_score(pred_trgs, trgs)### bleu_score = calculate_bleu(test_data, SRC, TRG, model, DEVICE)## print(f'BLEU score = {bleu_score * 100:.2f}')### def translate_sentence_vectorized(src_tensor, src_field, trg_field, model, device, max_len=50):#     assert isinstance(src_tensor, torch.Tensor)##     model.eval()#     src_mask = model.make_src_mask(src_tensor)##     with torch.no_grad():#         enc_src = model.encoder(src_tensor, src_mask)#     # enc_src = [batch_sz, src_len, hid_dim]##     trg_indexes = [[trg_field.vocab.stoi[trg_field.init_token]] for _ in range(len(src_tensor))]#     # Even though some examples might have been completed by producing a <eos> token#     # we still need to feed them through the model because other are not yet finished#     # and all examples act as a batch. Once every single sentence prediction encounters#     # <eos> token, then we can stop predicting.#     translations_done = [0] * len(src_tensor)#     for i in range(max_len):#         trg_tensor = torch.LongTensor(trg_indexes).to(device)#         trg_mask = model.make_trg_mask(trg_tensor)#         with torch.no_grad():#             output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)#         pred_tokens = output.argmax(2)[:, -1]#         for i, pred_token_i in enumerate(pred_tokens):#             trg_indexes[i].append(pred_token_i)#             if pred_token_i == trg_field.vocab.stoi[trg_field.eos_token]:#                 translations_done[i] = 1#         if all(translations_done):#             break##     # Iterate through each predicted example one by one;#     # Cut-off the portion including the after the <eos> token#     pred_sentences = []#     for trg_sentence in trg_indexes:#         pred_sentence = []#         for i in range(1, len(trg_sentence)):#             if trg_sentence[i] == trg_field.vocab.stoi[trg_field.eos_token]:#                 break#             pred_sentence.append(trg_field.vocab.itos[trg_sentence[i]])#         pred_sentences.append(pred_sentence)##     return pred_sentences, attention### from torchtext.data.metrics import bleu_score### def calculate_bleu_alt(iterator, src_field, trg_field, model, device, max_len=50):#     trgs = []#     pred_trgs = []#     with torch.no_grad():#         for batch in iterator:#             src = batch.src#             trg = batch.trg#             _trgs = []#             for sentence in trg:#                 tmp = []#                 # Start from the first token which skips the <start> token#                 for i in sentence[1:]:#                     # Targets are padded. So stop appending as soon as a padding or eos token is encountered#                     if i == trg_field.vocab.stoi[trg_field.eos_token] or i == trg_field.vocab.stoi[trg_field.pad_token]:#                         break#                     tmp.append(trg_field.vocab.itos[i])#                 _trgs.append([tmp])#             trgs += _trgs#             pred_trg, _ = translate_sentence_vectorized(src, src_field, trg_field, model, device)#             pred_trgs += pred_trg#     return pred_trgs, trgs, bleu_score(pred_trgs, trgs)