import argparse

def get_args(rest_args):
    parser = argparse.ArgumentParser(description='MAML-PPO-based ABR with vmaf')
    parser.add_argument('--test', action='store_true', help='Evaluate only')
    parser.add_argument('--name', default='pensieve', help='the name of algorithm')
    parser.add_argument('--a2c', action='store_true', help='Train policy with A2C')
    parser.add_argument('--ppo', action='store_true', help='Train policy with PPO')
    parser.add_argument('--a2br', action='store_true', help='Train policy with meta-ppo(a2br,Huang)')
    parser.add_argument('--agent-num', nargs='?', const=16, default=16, type=int, help='env numbers')
    parser.add_argument('--valid-i', nargs='?', const=1000, default=1000, type=int, help='checkpoint')
    parser.add_argument('--proba', action='store_true', help='Use probabilistic policy')
    parser.add_argument('--init', action='store_true', help='Load the pre-train model parameters')

    ## --------- Env configuration ----------
    parser.add_argument('--ro-len', nargs='?', const=50, default=50, type=int, help='Length of roll-out')
    parser.add_argument('--scaling-lb', nargs='?', const=6, default=6, type=float, help='The lower bound of rewards')
    parser.add_argument('--scaling-r', nargs='?', const=100., default=100., type=float, help='The scaling factor of rewards')
    
    ## --------- MAML-PPO -----------
    ## Policy loss 
    parser.add_argument('--ent-coeff', nargs='?', const=0.5, default=0.5, type=float, help='The coefficient of entropy in the loss function')
    parser.add_argument('--ent-decay', nargs='?', const=0.99, default=0.99, type=float, help='The decay coefficient of entropy')
    parser.add_argument('--gae-gamma', nargs='?', const=0.99, default=0.99, type=float, help='The gamma coefficent for GAE estimation')
    parser.add_argument('--gae-lambda', nargs='?', const=0.95, default=0.95, type=float, help='The lambda coefficent for GAE estimation')
    parser.add_argument('--lr-adapt', nargs='?', const=1e-5, default=1e-5, type=float, help='The learning rate of fast adaptation/inner update')
    parser.add_argument('--lr-meta', nargs='?', const=1e-4, default=1e-4, type=float, help='The learning rate of outer update')
    parser.add_argument('--adapt-steps', nargs='?', const=4, default=4, type=int, help='The number of inner update steps')
    
    ## PPO configures 
    parser.add_argument('--batch-size', nargs='?', const=128, default=128, type=int, help='Minibatch size for training')
    parser.add_argument('--ppo-ups', nargs='?', const=2, default=2, type=int, help='Update numbers in each epoch for PPO')
    parser.add_argument('--clip', nargs='?', const=0.02, default=0.02, type=float, help='Clip value of ppo')
    parser.add_argument('--dual-adv-w', nargs='?', const=3, default=3, type=float, help='The weight of adv value in dual-clip ppo')
    
    ## choose datasets for throughput traces 
    parser.add_argument('--tf', action='store_true', help='Use FCC traces')
    parser.add_argument('--tfh', action='store_true', help='Use FCC_and_3GP traces')
    parser.add_argument('--to', action='store_true', help='Use Oboe traces')
    parser.add_argument('--t3g', action='store_true', help='Use 3GP traces')
    parser.add_argument('--tg', action='store_true', help='Use Ghent traces')
    parser.add_argument('--tn', action='store_true', help='Use FH-Noisy traces')
    parser.add_argument('--tp', action='store_true', help='Use Puffer traces')
    parser.add_argument('--tp2', action='store_true', help='Use Puffer2 traces')
    parser.add_argument('--tw', action='store_true', help='Use Wifi traces')
    parser.add_argument('--ti', action='store_true', help='Use intern traces')

    return parser.parse_args(rest_args)