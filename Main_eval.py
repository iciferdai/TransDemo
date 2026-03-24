from modelMgmt import *

def main_eval_manual():
    print('init model...')
    model = MyTransf()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Preparing data...')
    train_dataloader, test_dataloader = process_data()
    print('init ModelManagement...')
    m_mgmt = ModelManagement(model, train_dataloader, test_dataloader, dev)
    print('init evaluate...')
    m_mgmt.init_eval()
    print('load checkpoint...')
    m_mgmt.load_checkpoint('CheckPoint_Ep100_0.0464_0.0481.pth', True)
    time.sleep(0.01)
    input_t = input("\nPress send input: ")
    m_mgmt.predict_manual(input_t)

def main_eval_auto():
    print('init model...')
    model = MyTransf()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Preparing data...')
    train_dataloader, test_dataloader = process_data()
    print('init ModelManagement...')
    m_mgmt = ModelManagement(model, train_dataloader, test_dataloader, dev)
    print('init evaluate...')
    m_mgmt.init_eval()
    print('load checkpoint...')
    #m_mgmt.load_checkpoint('best_test.pth', True)
    m_mgmt.load_checkpoint('CheckPoint_Ep100_0.0464_0.0481.pth', True)
    time.sleep(0.01)
    m_mgmt.predict_auto()

if __name__ == '__main__':
    main_eval_manual()