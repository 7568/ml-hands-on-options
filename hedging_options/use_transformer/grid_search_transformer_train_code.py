import osimport sysimport torchimport torch.nn as nn# Append the library path to PYTHONPATH, so library can be imported.sys.path.append(os.path.dirname("../../*"))sys.path.append(os.path.dirname("../*"))from torch.utils.data import DataLoaderimport matplotlib.pyplot as pltimport matplotlib.ticker as tickerimport numpy as npimport pandas as pdimport randomimport timefrom hedging_options.library import datasetimport transformer_netimport transformer_train_code# from tqdm import tqdmimport logging# from torch.utils.tensorboard import SummaryWriter# a=torch.from_numpy(np.array([1, 1, 1, 1, 1, 0, 0])).bool()# print(a)# b = torch.tril(torch.ones((7, 7))).bool()# print(b)# trg_mask = a & b# print(trg_mask)N_EPOCHS = 100CLIP = 1# gpu_ids = '0,1,2'# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idsENC_LAYERS = 3DEVICE = 'cpu'INTPUT_DIM = 20HID_DIM = 56ENC_HEADS = 4ENC_DROPOUT = 0.1IS_TRAIN = TrueH_P_L_BS = [[0.000005, 96], [0.000005, 16], [0.000005, 64],            [0.00001, 76], [0.00001, 16], [0.00001, 64],            [0.00002, 8], [0.00002, 16], [0.00002, 64],            [0.00005, 8], [0.00005, 16], [0.00005, 64],            [0.0001, 8], [0.0001, 16], [0.0001, 64],            [0.0002, 8], [0.0002, 16], [0.0002, 64],            [0.0005, 8], [0.0005, 16], [0.0005, 64]]if len(sys.argv) > 1:    ENC_LAYERS = int(sys.argv[1])    # print(f'CUDA_VISIBLE_DEVICES : {sys.argv[2]}')    # torch.cuda.set_device(int(sys.argv[2]))    if torch.cuda.is_available():        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(sys.argv[2]))        DEVICE = torch.device(f'cuda:{str(int(sys.argv[2]))}')    else:        DEVICE = torch.device('cpu')    transformer_train_code.DEVICE = DEVICE    H_P_L_BS = H_P_L_BS[int(sys.argv[3]):int(sys.argv[4])]# 3,6,9,16# print(f'ENC_LAYERS : {ENC_LAYERS}')# python transformer-code-comments.py > 0.0005-log &# PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/'transformer_train_code.PREPARE_HOME_PATH = f'/home/zhanghu/liyu/data/'NUM_WORKERS = 3if __name__ == '__main__':    logger = logging.getLogger()    logger.setLevel(level=logging.DEBUG)    sys.stderr = open(f'log/{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}_{sys.argv[4]}_grid_search.log', 'a')    transformer_train_code.NUM_WORKERS = NUM_WORKERS    # for i in [3, 6, 9]:    #     ENC_LAYERS = i    for h_p_l_b in H_P_L_BS:        BEST_VALID_LOSS = float('inf')        LEARNING_RATE = h_p_l_b[0]        BATCH_SIZE = h_p_l_b[1]        train_dataloader, val_dataloader, test_dataloader, training_dataset_length, valid_dataset_length, \        test_dataset_length = transformer_train_code.load_data(BATCH_SIZE)        handler = logging.FileHandler(f'log/grid_search_{sys.argv[1]}_{LEARNING_RATE}_{BATCH_SIZE}.log')        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')        handler.setFormatter(formatter)        logger.addHandler(handler)        logger.debug(f'{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}')        # Create data loaders.        # writer = SummaryWriter()        enc = transformer_net.Encoder(INTPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_DROPOUT, DEVICE)        model = transformer_net.Seq2Seq(enc, INTPUT_DIM, DEVICE).to(DEVICE)        logger.debug(f'The model has {transformer_net.count_parameters(model):,} trainable parameters')        # model.apply(transformer_net.initialize_weights)        if os.path.exists(f'pt/{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}-model.pt'):            model.load_state_dict(torch.load(f'pt/{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}-model.pt'))            logger.debug(f'use pt/{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}-model.pt parameters')        else:            model.apply(transformer_net.initialize_weights)            logger.debug(f'use transformer_net.initialize_weights parameters')        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)        if IS_TRAIN:            for epoch in range(N_EPOCHS):                start_time = time.time()                train_loss = transformer_train_code.train(model, optimizer, CLIP, BATCH_SIZE,                                                          train_dataloader, training_dataset_length)                valid_loss = transformer_train_code.evaluate(model, BATCH_SIZE,                                                             val_dataloader, valid_dataset_length)                end_time = time.time()                epoch_mins, epoch_secs = transformer_train_code.epoch_time(start_time, end_time)                if valid_loss < BEST_VALID_LOSS:                    BEST_VALID_LOSS = valid_loss                    torch.save(model.state_dict(), f'pt/{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}-model.pt')                logger.debug(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')                logger.debug(f'Train Loss: {train_loss:.3f} ')                logger.debug(f'Validate Loss: {valid_loss:.3f} ')                if epoch % 3 == 0:                    logger.debug(f'MSHE in test : '                                 f''                                 f'{transformer_train_code.get_mshe_in_test(model, BATCH_SIZE, test_dataloader, test_dataset_length)}')        model.load_state_dict(torch.load(f'pt/{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}-model.pt'))        logger.debug(f'MSHE in test : '                     f'{transformer_train_code.get_mshe_in_test(model, BATCH_SIZE, test_dataloader, test_dataset_length)}')        logger.removeHandler(handler)        del train_dataloader        del val_dataloader        del test_dataloader        del optimizer        del enc        del model    # del logging# def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):#     model.eval()##     if isinstance(sentence, str):#         nlp = spacy.load('de_core_news_sm')#         tokens = [token.text.lower() for token in nlp(sentence)]#     else:#         tokens = [token.lower() for token in sentence]##     tokens = [src_field.init_token] + tokens + [src_field.eos_token]##     src_indexes = [src_field.vocab.stoi[token] for token in tokens]##     src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)##     src_mask = model.make_src_mask(src_tensor)##     with torch.no_grad():#         enc_src = model.encoder(src_tensor, src_mask)##     trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]##     for i in range(max_len):##         trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)##         trg_mask = model.make_trg_mask(trg_tensor)##         with torch.no_grad():#             output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)##         pred_token = output.argmax(2)[:, -1].item()##         trg_indexes.append(pred_token)##         if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:#             break##     trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]##     return trg_tokens[1:], attention### def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):#     assert n_rows * n_cols == n_heads##     fig = plt.figure(figsize=(15, 25))##     for i in range(n_heads):#         ax = fig.add_subplot(n_rows, n_cols, i + 1)##         _attention = attention.squeeze(0)[i].cpu().detach().numpy()##         cax = ax.matshow(_attention, cmap='bone')##         ax.tick_params(labelsize=12)#         ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],#                            rotation=45)#         ax.set_yticklabels([''] + translation)##         ax.xaxis.set_major_locator(ticker.MultipleLocator(1))#         ax.yaxis.set_major_locator(ticker.MultipleLocator(1))##     plt.show()#     plt.close()### example_idx = 8## src = vars(train_data_map.examples[example_idx])['src']# trg = vars(train_data_map.examples[example_idx])['trg']## print(f'src = {src}')# print(f'trg = {trg}')## translation, attention = translate_sentence(src, SRC, TRG, model, DEVICE)## print(f'predicted trg = {translation}')## display_attention(src, translation, attention)## example_idx = 6## src = vars(valid_data.examples[example_idx])['src']# trg = vars(valid_data.examples[example_idx])['trg']## print(f'src = {src}')# print(f'trg = {trg}')# translation, attention = translate_sentence(src, SRC, TRG, model, DEVICE)## print(f'predicted trg = {translation}')## display_attention(src, translation, attention)## example_idx = 10## src = vars(test_data.examples[example_idx])['src']# trg = vars(test_data.examples[example_idx])['trg']## print(f'src = {src}')# print(f'trg = {trg}')## translation, attention = translate_sentence(src, SRC, TRG, model, DEVICE)## print(f'predicted trg = {translation}')## display_attention(src, translation, attention)### def calculate_bleu(data, src_field, trg_field, model, device, max_len=50):#     trgs = []#     pred_trgs = []##     for datum in data:#         src = vars(datum)['src']#         trg = vars(datum)['trg']##         pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)##         # cut off <eos> token#         pred_trg = pred_trg[:-1]##         pred_trgs.append(pred_trg)#         trgs.append([trg])##     return bleu_score(pred_trgs, trgs)### bleu_score = calculate_bleu(test_data, SRC, TRG, model, DEVICE)## print(f'BLEU score = {bleu_score * 100:.2f}')### def translate_sentence_vectorized(src_tensor, src_field, trg_field, model, device, max_len=50):#     assert isinstance(src_tensor, torch.Tensor)##     model.eval()#     src_mask = model.make_src_mask(src_tensor)##     with torch.no_grad():#         enc_src = model.encoder(src_tensor, src_mask)#     # enc_src = [batch_sz, src_len, hid_dim]##     trg_indexes = [[trg_field.vocab.stoi[trg_field.init_token]] for _ in range(len(src_tensor))]#     # Even though some examples might have been completed by producing a <eos> token#     # we still need to feed them through the model because other are not yet finished#     # and all examples act as a batch. Once every single sentence prediction encounters#     # <eos> token, then we can stop predicting.#     translations_done = [0] * len(src_tensor)#     for i in range(max_len):#         trg_tensor = torch.LongTensor(trg_indexes).to(device)#         trg_mask = model.make_trg_mask(trg_tensor)#         with torch.no_grad():#             output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)#         pred_tokens = output.argmax(2)[:, -1]#         for i, pred_token_i in enumerate(pred_tokens):#             trg_indexes[i].append(pred_token_i)#             if pred_token_i == trg_field.vocab.stoi[trg_field.eos_token]:#                 translations_done[i] = 1#         if all(translations_done):#             break##     # Iterate through each predicted example one by one;#     # Cut-off the portion including the after the <eos> token#     pred_sentences = []#     for trg_sentence in trg_indexes:#         pred_sentence = []#         for i in range(1, len(trg_sentence)):#             if trg_sentence[i] == trg_field.vocab.stoi[trg_field.eos_token]:#                 break#             pred_sentence.append(trg_field.vocab.itos[trg_sentence[i]])#         pred_sentences.append(pred_sentence)##     return pred_sentences, attention### from torchtext.data.metrics import bleu_score### def calculate_bleu_alt(iterator, src_field, trg_field, model, device, max_len=50):#     trgs = []#     pred_trgs = []#     with torch.no_grad():#         for batch in iterator:#             src = batch.src#             trg = batch.trg#             _trgs = []#             for sentence in trg:#                 tmp = []#                 # Start from the first token which skips the <start> token#                 for i in sentence[1:]:#                     # Targets are padded. So stop appending as soon as a padding or eos token is encountered#                     if i == trg_field.vocab.stoi[trg_field.eos_token] or i == trg_field.vocab.stoi[trg_field.pad_token]:#                         break#                     tmp.append(trg_field.vocab.itos[i])#                 _trgs.append([tmp])#             trgs += _trgs#             pred_trg, _ = translate_sentence_vectorized(src, src_field, trg_field, model, device)#             pred_trgs += pred_trg#     return pred_trgs, trgs, bleu_score(pred_trgs, trgs)