# Equipment Pose Estimator Test Loop
 ![test_loop](https://user-images.githubusercontent.com/50568142/142012721-aae368d6-1160-437a-b190-01b6b9705fea.png)</br>
 A test loop that trains and evaluates models along preset test settings procedurally.</br>

## Getting started

### 1. Clone the repository
 ```
 git clone https://github.com/bigbreadguy/pose-estimator-test-loop.git
 ```

### 2. Install all the requirements
 ```
 pip install -r requirements.txt
 ```
 </br>
 **You need to install pyTorch version>=1.10.0**
 </br>

### 3. Run the script
 ```
 python main.py [--loop {"stroll, "train_test", "test", "evaluate"}]
                [--lr {FLOAT}}] [--batch_size {INT}] [--train_continue {"on", "off}]
                [--num_epoch {INT}] [--task {"pose estimation"}]
                [--ny {INT}] [--nx {INT}] [--nch {INT}] [--nker {INT}]
                [--norm {"inorm", "bnorm"}] [--network {"PoseResNet", "PoseResNetv2"}]
                [--resnet_depth {18, 34, 50, 101, 152}] [--cuda {"cuda", "cuda:0", "cuda:1"}]
                [--spec {"all" or ANY-TEST-DESIGN-NAMES}]
 ```
 </br>
 
##### Arguments explained
 **loop :** stroll is recommended if you set as other settings, then the loop strolls for only the setting.</br>
 **lr :** defines learning rate, default value is 2e-4.</br>
 **batch_size :** defines batch size, default is 4, yet 2 is recommended.</br>
 **train_continue :** defines whether the pre-trained model should be loaded or not.</br>
 **num_epoch :** defines maximum epochs for training and just give a big number, early stopping is supported.</br>
 **ny :** height-wise dimension of given images.</br>
 **nx :** width-wise dimension of given images.</br>
 **nch :** channel-wise dimension of given images.</br>
 **nker :** base numbers of kernels in the ResNet blocks, 64 is default.</br>
 **norm :** defines what kind of normalization method should be attatched</br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bnorm : batch normalization,</br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;inorm : instance normalization.</br>
 **network :** defines the network that will be served as pose estimator. PoseResNetv2 initilizes the model with ImageNet pretrained parameters.</br>
 **resnet_depth :** defines how many layers the ResNet will have.</br>
 **cuda :** you can select the device with this argument.</br>
 **spec :** an optional argument for re-try a train loop for designated test setting.</br>

### 4. The test loop will stroll along test settings as shown
 ![test_setting](https://user-images.githubusercontent.com/50568142/142012733-42e9cf32-2f75-43d1-92dc-e85a53b1a63c.png)
