import argparse
parser = argparse.ArgumentParser(description='Test Change Detection Models')

####------------------------------------   ttsting parameters   --------------------------------------####

parser.add_argument('--test_batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--num_workers', default=0, type=int, help='num_workers')
parser.add_argument('--gpu_id', default="0", type=str, help='which gpu to run.')
parser.add_argument('--suffix', default=['.png','.jpg','.tif'], type=list, help='the suffix of the image files.')


####----------------------------------   path for loading data   -----------------------------------------####
parser.add_argument('--test1_dir', default='./LEVIR-CD256/test/A', type=str, help='t1 image path for testing')
parser.add_argument('--test2_dir', default='./LEVIR-CD256/test/B', type=str, help='t2 image path for testing')
parser.add_argument('--label_test_ori', default='./LEVIR-CD256/test/label', type=str, help='label path for testing')
parser.add_argument('--label_test_1_2', default='./LEVIR-CD256/test/labelds(1_2)', type=str, help='label path for testing')
parser.add_argument('--label_test_1_4', default='./LEVIR-CD256/test/labelds(1_4)', type=str, help='label path for testing')
parser.add_argument('--label_test_1_8', default='./LEVIR-CD256/test/labelds(1_8)', type=str, help='label path for testing')


####----------------------------   network loading and result saving   ------------------------------------####
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='models are saved here')
parser.add_argument('--name', type=str, default='LEVIR', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--load_net', type=str, default='HMLNet_levir.pth',
                        help='name of the experiment. It decides where to store samples and models') 
parser.add_argument('--results_dir', type=str, default='./result/LEVIR/HMLNet', help='saves results here.')
parser.add_argument('--alpha', type=int, default=0.3, help='the balance weight of BCL and Dice. alpha=1, the total loss is Dice. alpha=0, the total loss is BCL.')

####-------------------------------------   Model settings   -----------------------------------------####
parser.add_argument('--in_c', default=3, type=int, help='input channel')
parser.add_argument('--out_c', default=128, type=int, help='Backbone output channel')
parser.add_argument('--out_cls', default=2, type=int, help='output category')