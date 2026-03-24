from modelMgmt import *

def main_train():
    print('init model...')
    model = MyTransf()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Preparing data...')
    train_dataloader, test_dataloader = process_data()
    print('init ModelManagement...')
    m_mgmt = ModelManagement(model, train_dataloader, test_dataloader, dev)
    print('init train...')
    m_mgmt.init_train()
    m_mgmt.init_dashboard()
    print('Start train...')
    m_mgmt.train_epochs(100)
    m_mgmt.save_state()
    m_mgmt.show_dashboard()

def check_status():
    print('init model & ModelManagement...')
    model = MyTransf()
    train_dataloader, test_dataloader = None, None
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m_mgmt = ModelManagement(model, train_dataloader, test_dataloader, dev)
    print('load status of best_test...')
    m_mgmt.load_checkpoint('best_test_bak.pth', True)
    m_mgmt.progress_info(True)
    print('load status of CheckPoint_Ep1000...')
    m_mgmt.load_checkpoint('CheckPoint_Ep1000_0.0281_0.0581.pth')
    m_mgmt.progress_info(True)
    print('load status of state...')
    m_mgmt.init_dashboard()
    m_mgmt.load_state('State_Ep1000_0.0323.pkl')
    m_mgmt.update_dashboard()
    m_mgmt.show_dashboard()

if __name__ == '__main__':
    main_train()
    #check_status()