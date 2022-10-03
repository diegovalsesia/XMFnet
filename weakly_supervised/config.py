



class params:
    def __init__(self): 
        
        #General Parameters
        self.cat = "all"
        
        
        #DGCNN parameters
        self.emb_dims = 256
        self.dropout = 0.5
        self.k = 20 #number of NN to select when constructing the graph
        
        #EdgePooling parameters
        self.k_pool1 = 16
        self.k_pool2 = 6
        self.pool1_points = 512
        self.pool2_points = 128
        self.scoring_fun = "tanh"

        #Decoder parameters 
        self.num_branch = 8
        self.K1 = 64
        self.K2 = 64
        self.N = 128
        self.method = "integrated"
        self.alpha = 0.01


        #Multi-head Attention parameters
        self.d_attn = 256
        self.num_heads = 4

        #Training parameters
        self.batch_size = 32
        self.nThreads = 1
        self.lr = 0.0001
        self.dataroot = "/datapath"
        self.n_epochs = 200
        self.ckp_dir = "/checkpoint_path"
        self.ckp_epoch = 5
        self.eval_epoch = 2
        self.resume = ''
        self.loss_print = 500
        self.lambd = 0.2
        self.vis_step = 1


        #Renderer
        self.render_size = 224
        self.radius = 0.025
        self.points_per_pixel = 16
        self.clip_pts_grad = 0.05
        self.image_size = 224
        self.dist = 2.5

        #Completion self
        self.n_points = 2048
        self.mixup_params = 1.0
        