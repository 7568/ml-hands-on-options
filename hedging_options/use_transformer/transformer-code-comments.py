import osimport sysimport torchimport torch.nn as nn# Append the library path to PYTHONPATH, so library can be imported.sys.path.append(os.path.dirname(os.getcwd()))from torch.utils.data import DataLoaderimport matplotlib.pyplot as pltimport matplotlib.ticker as tickerimport numpy as npimport pandas as pdimport randomimport timefrom hedging_options.library import dataset# from tqdm import tqdmimport logginglogger = logging.getLogger()logger.setLevel(level=logging.DEBUG)# from torch.utils.tensorboard import SummaryWriter# a=torch.from_numpy(np.array([1, 1, 1, 1, 1, 0, 0])).bool()# print(a)# b = torch.tril(torch.ones((7, 7))).bool()# print(b)# trg_mask = a & b# print(trg_mask)SEED = 1234random.seed(SEED)np.random.seed(SEED)torch.manual_seed(SEED)torch.cuda.manual_seed(SEED)torch.backends.cudnn.deterministic = Trueclass Encoder(nn.Module):    def __init__(self, input_dim, hid_dim, n_layers, n_heads, dropout, device, max_length=100):        super().__init__()        self.device = device        # self.pos_embedding = nn.Embedding(max_length, hid_dim)        self.pos = None        self.layers = nn.ModuleList([EncoderLayer(input_dim, hid_dim, n_heads, dropout, device) for _ in range(            n_layers)])        self.dropout = nn.Dropout(dropout)        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)    def forward(self, src):        # src = [batch size, src len]        # src_mask = [batch size, 1, 1, src len]        # batch_size = src.shape[0]        # src_len = src.shape[1]        # src_feature_len = src.shape[2]        # if self.pos is None:        # self.pos = torch.arange(0, src_len).repeat(src_feature_len, 1).transpose(0, 1).repeat(batch_size, 1,        #                                                                                       1).float().to(self.device)        # pos = [batch size, src len]        # src = self.dropout(src * self.scale + self.pos)        # src = [batch size, src len, hid dim]        for layer in self.layers:            src = layer(src)        # src = [batch size, src len, hid dim]        return srcclass EncoderLayer(nn.Module):    def __init__(self, input_dim, hid_dim, n_heads, dropout, device):        super().__init__()        self.self_attn_layer_norm = nn.LayerNorm(input_dim)        self.ff_layer_norm = nn.LayerNorm(input_dim)        self.self_attention = MultiHeadAttentionLayer(input_dim, hid_dim, n_heads, dropout, device)        # self.self_attention_2 = MultiHeadAttentionLayer(hid_dim,input_dim, n_heads, dropout, device)        # self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)        self.dropout = nn.Dropout(dropout)    def forward(self, src):        # src = [batch size, src len, hid dim]        # src_mask = [batch size, 1, 1, src len]        # self attention        # print(src.shape)        _src, _ = self.self_attention(src, src, src)        # __src, __ = self.self_attention_2(_src, _src, _src)        # dropout, residual connection and layer norm        src = self.self_attn_layer_norm(src + self.dropout(_src))        # src = [batch size, src len, hid dim]        # positionwise feedforward        # _src = self.positionwise_feedforward(src)        # dropout, residual and layer norm        # src = self.ff_layer_norm(src + self.dropout(_src))        # src = self.ff_layer_norm(src)        # src = [batch size, src len, hid dim]        return srcclass MultiHeadAttentionLayer(nn.Module):    def __init__(self, input_dim, hid_dim, n_heads, dropout, device):        super().__init__()        assert hid_dim % n_heads == 0        self.hid_dim = hid_dim        self.n_heads = n_heads        self.head_dim = hid_dim // n_heads        self.fc_q = nn.Linear(input_dim, hid_dim)        self.fc_k = nn.Linear(input_dim, hid_dim)        self.fc_v = nn.Linear(input_dim, hid_dim)        self.fc_o = nn.Linear(hid_dim, input_dim)        self.dropout = nn.Dropout(dropout)        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)    def forward(self, query, key, value, mask=None):        batch_size = query.shape[0]        # query = [batch size, query len, hid dim]        # key = [batch size, key len, hid dim]        # value = [batch size, value len, hid dim]        Q = self.fc_q(query)        K = self.fc_k(key)        V = self.fc_v(value)        # Q = [batch size, query len, hid dim]        # K = [batch size, key len, hid dim]        # V = [batch size, value len, hid dim]        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)        # Q = [batch size, n heads, query len, head dim]        # K = [batch size, n heads, key len, head dim]        # V = [batch size, n heads, value len, head dim]        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale        # energy = [batch size, n heads, query len, key len]        attention = torch.softmax(energy, dim=-1)        # attention = [batch size, n heads, query len, key len]        x = torch.matmul(self.dropout(attention), V)        # x = [batch size, n heads, query len, head dim]        x = x.permute(0, 2, 1, 3).contiguous()        # x = [batch size, query len, n heads, head dim]        x = x.view(batch_size, -1, self.hid_dim)        # x = [batch size, query len, hid dim]        x = self.fc_o(x)        # x = [batch size, query len, hid dim]        return x, attention# class PositionwiseFeedforwardLayer(nn.Module):#     def __init__(self, hid_dim, pf_dim, dropout):#         super().__init__()##         self.fc_1 = nn.Linear(hid_dim, pf_dim)#         self.fc_2 = nn.Linear(pf_dim, hid_dim)##         self.dropout = nn.Dropout(dropout)##     def forward(self, x):#         # x = [batch size, seq len, hid dim]##         x = self.dropout(torch.relu(self.fc_1(x)))##         # x = [batch size, seq len, pf dim]##         x = self.fc_2(x)##         # x = [batch size, seq len, hid dim]##         return xclass Seq2Seq(nn.Module):    def __init__(self, encoder, input_dim, device):        super().__init__()        self.encoder = encoder        # self.decoder = None        # self.src_pad_idx = src_pad_idx        # self.trg_pad_idx = trg_pad_idx        self.device = device        self.fc_o = nn.Linear(input_dim, 1)        self.fc_o2 = nn.Linear(15, 1)    def make_src_mask(self, src):        # src = [batch size, src len]        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)        # src_mask = [batch size, 1, 1, src len]        return src_mask    def make_trg_mask(self, trg):        # trg = [batch size, trg len]        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)        # trg_pad_mask = [batch size, 1, 1, trg len]        trg_len = trg.shape[1]        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()        # trg_sub_mask = [trg len, trg len]        trg_mask = trg_pad_mask & trg_sub_mask        # trg_mask = [batch size, 1, trg len, trg len]        return trg_mask    def forward(self, src, results):        # src = [batch size, src len]        # trg = [batch size, trg len]        # src_mask = self.make_src_mask(src)        # trg_mask = self.make_trg_mask(trg)        # src_mask = [batch size, 1, 1, src len]        # trg_mask = [batch size, 1, trg len, trg len]        output_ = torch.relu(self.fc_o(self.encoder(src)))        output_dim = output_.shape[0]        output = self.fc_o2(output_.view(output_dim, -1))        if torch.isnan(output).any():            print('\nweight error')        # print(output.shape)        # v_1 = output.contiguous().view(output_dim)        i = 0        for t in src:            i += 1            if torch.isnan(t).any():                print(i)        R_1 = results.contiguous().view(output_dim, -1)        # print(output.shape)        # print(R_1.shape)        # output = output - R_1[:, 1].view(output_dim, -1)        # print(R_1[:, 1].shape)        # print(output.shape)        s_1 = R_1[:, -1].view(output_dim, -1)        one_ret = 1 + R_1[:, 0].view(output_dim, -1) / 253        s_0 = R_1[:, 3].view(output_dim, -1)        c_0 = R_1[:, 2].view(output_dim, -1)        output = output * s_1 + one_ret * (c_0 - output * s_0)        # print(output.shape)        # enc_src = [batch size, src len, hid dim]        # output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)        # output = [batch size, trg len, output dim]        # attention = [batch size, n heads, trg len, src len]        return outputdef count_parameters(model):    return sum(p.numel() for p in model.parameters() if p.requires_grad)def initialize_weights(m):    if hasattr(m, 'weight') and m.weight.dim() > 1:        nn.init.xavier_uniform_(m.weight.data)def train(model, optimizer, criterion, clip, batch_size):    model.train()    all_loss = []    # for ii, (datas, results) in tqdm(enumerate(train_dataloader), total=len(train_data_index) / batch_size):    for ii, (datas, results) in enumerate(train_dataloader):        datas = datas.float().to(DEVICE)        results = results.float().to(DEVICE)        output = model(datas, results)        if torch.isnan(output).any():            print(ii)        output_dim = output.shape[0]        R_1 = results.contiguous().view(output_dim, -1)        y = R_1[:, -2].view(output_dim, -1)        # print(y-y_hat)        # v_1 = output.contiguous().view(-1, output_dim)[:, -1]        # trg = results.contiguous().view(-1)        loss = criterion(output, y)        # print(output,' ===== ',y)        # print(loss.item())        optimizer.zero_grad()        loss.backward()        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)        optimizer.step()        # print('=============================')        # for name, parameters in model.named_parameters():        #     if name == 'encoder.layers.0.self_attention.fc_q.weight':        #         print('\n')        #         print(parameters.grad.data[0,:5],'       ',parameters.data[0,:5])        # print(name, ':', parameters.size(),' : ', parameters.grad)        # print('=============================')        # epoch_loss += loss.item()        all_loss.append(loss.item())        # print(loss.item())    return np.array(all_loss).mean()def evaluate(model, criterion, batch_size=1):    model.eval()    # epoch_loss = 0    all_loss = []    with torch.no_grad():        # for ii, (datas, results) in tqdm(enumerate(val_dataloader), total=len(valid_data_index) / batch_size):        for ii, (datas, results) in enumerate(val_dataloader):            datas = datas.float().to(DEVICE)            results = results.float().to(DEVICE)            optimizer.zero_grad()            output = model(datas, results)            output_dim = output.shape[0]            R_1 = results.contiguous().view(output_dim, -1)            y = R_1[:, -2].view(output_dim, -1)            # print(y-y_hat)            # v_1 = output.contiguous().view(-1, output_dim)[:, -1]            # trg = results.contiguous().view(-1)            loss = criterion(output, y)            # epoch_loss += loss.item()            all_loss.append(loss.item())            # if math.isnan(epoch_loss):            #     print(epoch_loss)    return np.array(all_loss).mean()def get_mshe_in_test(model, batch_size):    model.eval()    mshes_put = []    mshes_call = []    with torch.no_grad():        for ii, (datas, results) in enumerate(test_dataloader):            datas = datas.float().to(DEVICE)            results = results.float().to(DEVICE)            optimizer.zero_grad()            output = model(datas, results)            output_dim = output.shape[0]            R_1 = results.contiguous().view(output_dim, -1)            y = R_1[:, -2].view(output_dim, -1)            # v_1 = output.contiguous().view(-1, output_dim)[:, -1]            # trg = results.contiguous().view(-1)            mshe = torch.pow((100 * (output - y)) / R_1[:, -1], 2).detach().cpu().numpy()            # mshes = np.append(mshes, mshe)            if results[0, 1] == 1:                mshes_put = np.append(mshes_put, mshe)            else:                mshes_call = np.append(mshes_call, mshe)    return mshes_put.mean(),mshes_call.mean()def epoch_time(start_time, end_time):    elapsed_time = end_time - start_time    elapsed_mins = int(elapsed_time / 60)    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))    return elapsed_mins, elapsed_secsN_EPOCHS = 10CLIP = 1# gpu_ids = '0,1,2'# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idsENC_LAYERS = 3DEVICE = 'cpu'INTPUT_DIM = 19HID_DIM = 56ENC_HEADS = 4ENC_DROPOUT = 0.1IS_TRAIN = TrueH_P_L_BS = [[[0.0001, 8], [0.0001, 16], [0.0001, 64]],            [[0.0002, 8], [0.0002, 16], [0.0002, 64]],            [[0.0005, 8], [0.0005, 16], [0.0005, 64]],            [[0.001, 8], [0.001, 16], [0.001, 64]],            [[0.002, 8], [0.002, 16], [0.002, 64]]]if len(sys.argv) > 1:    ENC_LAYERS = int(sys.argv[1])    # print(f'CUDA_VISIBLE_DEVICES : {sys.argv[2]}')    # torch.cuda.set_device(int(sys.argv[2]))    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(sys.argv[2])+1)    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    H_P_L_BS=H_P_L_BS[int(sys.argv[2])]# 3,6,9,16# print(f'ENC_LAYERS : {ENC_LAYERS}')# python transformer-code-comments.py > 0.0005-log &# PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/china-market/'PREPARE_HOME_PATH = f'/home/zhanghu/liyu/data/'NUM_WORKERS = 10if __name__ == '__main__':    train_data_index = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300_training_index.csv').to_numpy()    valid_data_index = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300_validation_index.csv').to_numpy()    test_data_index = pd.read_csv(f'{PREPARE_HOME_PATH}/h_sh_300_test_index.csv').to_numpy()    # train_data_index = train_data_index[10000:, :]    # valid_data_index = valid_data_index[:3000, :]    training_dataset = dataset.Dataset_transformer(train_data_index, f'{PREPARE_HOME_PATH}/parquet/training/')    valid_dataset = dataset.Dataset_transformer(valid_data_index, f'{PREPARE_HOME_PATH}/parquet/validation/')    test_dataset = dataset.Dataset_transformer(test_data_index, f'{PREPARE_HOME_PATH}/parquet/test/')    for h_p_l_b in H_P_L_BS:        BEST_VALID_LOSS = float('inf')        LEARNING_RATE = h_p_l_b[0]        BATCH_SIZE = h_p_l_b[1]        handler = logging.FileHandler(f'log/{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}.log')        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')        handler.setFormatter(formatter)        logger.addHandler(handler)        logger.debug(f'{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}')        # Create data loaders.        train_dataloader = DataLoader(training_dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)        val_dataloader = DataLoader(valid_dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)        test_dataloader = DataLoader(test_dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)        # writer = SummaryWriter()        enc = Encoder(INTPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_DROPOUT, DEVICE)        model = Seq2Seq(enc, INTPUT_DIM, DEVICE).to(DEVICE)        logger.debug(f'The model has {count_parameters(model):,} trainable parameters')        model.apply(initialize_weights)        # model = torch.nn.DataParallel(model)        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)        criterion = nn.MSELoss()        if IS_TRAIN:            for epoch in range(N_EPOCHS):                start_time = time.time()                train_loss = train(model, optimizer, criterion, CLIP, BATCH_SIZE)                valid_loss = evaluate(model, criterion, BATCH_SIZE)                end_time = time.time()                epoch_mins, epoch_secs = epoch_time(start_time, end_time)                if valid_loss < BEST_VALID_LOSS:                    BEST_VALID_LOSS = valid_loss                    torch.save(model.state_dict(), f'pt/{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}-model.pt')                logger.debug(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')                logger.debug(f'Train Loss: {train_loss:.3f} ')                # print(f'Train PPL: {math.exp(train_loss):7.3f}')                # print(f'Val. Loss: {valid_loss:.3f}')                # print(f'Val. PPL: {math.exp(valid_loss):7.3f}')                # writer.add_scalar('Train Loss', f'{train_loss:.3f}')                # writer.add_scalar('Train PPL', f'{math.exp(train_loss):7.3f}')                # writer.add_scalar('Val. Loss', f'{valid_loss:.3f}', writer.count)                # writer.add_scalar('Val. PPL', f'{math.exp(valid_loss):7.3f}')        model.load_state_dict(torch.load(f'pt/{ENC_LAYERS}-{LEARNING_RATE}-{BATCH_SIZE}-model.pt'))        logger.debug(f'MSHE in test : {get_mshe_in_test(model, BATCH_SIZE)}')        logger.removeHandler(handler)        del train_dataloader        del val_dataloader        del test_dataloader        del optimizer        del criterion        del enc        del model        # del loggingdef translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):    model.eval()    if isinstance(sentence, str):        nlp = spacy.load('de_core_news_sm')        tokens = [token.text.lower() for token in nlp(sentence)]    else:        tokens = [token.lower() for token in sentence]    tokens = [src_field.init_token] + tokens + [src_field.eos_token]    src_indexes = [src_field.vocab.stoi[token] for token in tokens]    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)    src_mask = model.make_src_mask(src_tensor)    with torch.no_grad():        enc_src = model.encoder(src_tensor, src_mask)    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]    for i in range(max_len):        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)        trg_mask = model.make_trg_mask(trg_tensor)        with torch.no_grad():            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)        pred_token = output.argmax(2)[:, -1].item()        trg_indexes.append(pred_token)        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:            break    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]    return trg_tokens[1:], attentiondef display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):    assert n_rows * n_cols == n_heads    fig = plt.figure(figsize=(15, 25))    for i in range(n_heads):        ax = fig.add_subplot(n_rows, n_cols, i + 1)        _attention = attention.squeeze(0)[i].cpu().detach().numpy()        cax = ax.matshow(_attention, cmap='bone')        ax.tick_params(labelsize=12)        ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],                           rotation=45)        ax.set_yticklabels([''] + translation)        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))    plt.show()    plt.close()example_idx = 8src = vars(train_data_map.examples[example_idx])['src']trg = vars(train_data_map.examples[example_idx])['trg']print(f'src = {src}')print(f'trg = {trg}')translation, attention = translate_sentence(src, SRC, TRG, model, DEVICE)print(f'predicted trg = {translation}')display_attention(src, translation, attention)example_idx = 6src = vars(valid_data.examples[example_idx])['src']trg = vars(valid_data.examples[example_idx])['trg']print(f'src = {src}')print(f'trg = {trg}')translation, attention = translate_sentence(src, SRC, TRG, model, DEVICE)print(f'predicted trg = {translation}')display_attention(src, translation, attention)example_idx = 10src = vars(test_data.examples[example_idx])['src']trg = vars(test_data.examples[example_idx])['trg']print(f'src = {src}')print(f'trg = {trg}')translation, attention = translate_sentence(src, SRC, TRG, model, DEVICE)print(f'predicted trg = {translation}')display_attention(src, translation, attention)def calculate_bleu(data, src_field, trg_field, model, device, max_len=50):    trgs = []    pred_trgs = []    for datum in data:        src = vars(datum)['src']        trg = vars(datum)['trg']        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)        # cut off <eos> token        pred_trg = pred_trg[:-1]        pred_trgs.append(pred_trg)        trgs.append([trg])    return bleu_score(pred_trgs, trgs)bleu_score = calculate_bleu(test_data, SRC, TRG, model, DEVICE)print(f'BLEU score = {bleu_score * 100:.2f}')def translate_sentence_vectorized(src_tensor, src_field, trg_field, model, device, max_len=50):    assert isinstance(src_tensor, torch.Tensor)    model.eval()    src_mask = model.make_src_mask(src_tensor)    with torch.no_grad():        enc_src = model.encoder(src_tensor, src_mask)    # enc_src = [batch_sz, src_len, hid_dim]    trg_indexes = [[trg_field.vocab.stoi[trg_field.init_token]] for _ in range(len(src_tensor))]    # Even though some examples might have been completed by producing a <eos> token    # we still need to feed them through the model because other are not yet finished    # and all examples act as a batch. Once every single sentence prediction encounters    # <eos> token, then we can stop predicting.    translations_done = [0] * len(src_tensor)    for i in range(max_len):        trg_tensor = torch.LongTensor(trg_indexes).to(device)        trg_mask = model.make_trg_mask(trg_tensor)        with torch.no_grad():            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)        pred_tokens = output.argmax(2)[:, -1]        for i, pred_token_i in enumerate(pred_tokens):            trg_indexes[i].append(pred_token_i)            if pred_token_i == trg_field.vocab.stoi[trg_field.eos_token]:                translations_done[i] = 1        if all(translations_done):            break    # Iterate through each predicted example one by one;    # Cut-off the portion including the after the <eos> token    pred_sentences = []    for trg_sentence in trg_indexes:        pred_sentence = []        for i in range(1, len(trg_sentence)):            if trg_sentence[i] == trg_field.vocab.stoi[trg_field.eos_token]:                break            pred_sentence.append(trg_field.vocab.itos[trg_sentence[i]])        pred_sentences.append(pred_sentence)    return pred_sentences, attentionfrom torchtext.data.metrics import bleu_scoredef calculate_bleu_alt(iterator, src_field, trg_field, model, device, max_len=50):    trgs = []    pred_trgs = []    with torch.no_grad():        for batch in iterator:            src = batch.src            trg = batch.trg            _trgs = []            for sentence in trg:                tmp = []                # Start from the first token which skips the <start> token                for i in sentence[1:]:                    # Targets are padded. So stop appending as soon as a padding or eos token is encountered                    if i == trg_field.vocab.stoi[trg_field.eos_token] or i == trg_field.vocab.stoi[trg_field.pad_token]:                        break                    tmp.append(trg_field.vocab.itos[i])                _trgs.append([tmp])            trgs += _trgs            pred_trg, _ = translate_sentence_vectorized(src, src_field, trg_field, model, device)            pred_trgs += pred_trg    return pred_trgs, trgs, bleu_score(pred_trgs, trgs)