# Equipment Pose Estimator Test Loop
 ![test_loop](https://user-images.githubusercontent.com/50568142/139358613-e8dd6902-0d30-4183-9b1e-81e4a9967385.png)</br>
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

### 3. Run the script
 ```
 python main.py [--lr {FLOAT}}] [--batch_size {INT}] [--base_epoch {INT}]
                [--epoch_d {INT}] [--epoch_steps {INT}] [--task {"pose estimation"}]
                [--ny {INT}] [--nx {INT}] [--nch {INT}] [--nker {INT}]
                [--norm {"inorm", "bnorm"}] [--network {"PoseResNet"}]
                [--resnet_depth {18, 34, 50, 101, 152}] [--cuda {"cuda", "cuda:0", "cuda:1"}]
 ```
 </br>

 **lr** defines learning rate, default value is 2e-4.</br>
 **batch_size** defines batch size, default is 4, yet 2 is recommended.</br>
 
 **These 3 arguments defines grid search settings for the epochs.**</br>
 **base_epoch** defines very first epoch grid, the test loop will train the model until it reaches the value.</br>
 **epoch_d** defines displacement between the grids.</br>
 **epoch_steps** defines how many grids will be set.</br>

 **ny** height-wise dimension of given images.</br>
 **nx** width-wise dimension of given images.</br>
 **nch** channel-wise dimension of given images.</br>
 **nker** base numbers of kernels in the ResNet blocks, 64 is default.</br>
 **norm** defines what kind of normalization method should be attatched</br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bnorm : batch normalization,</br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;inorm : instance normalization.</br>
 **network** defines the network that will be served as pose estimator.</br>
 **resnet_depth** defines how many layers the ResNet will have.</br>
 **cuda** you can select the device with this argument.</br>

### 4. The test loop will stroll along test settings as shown
 ![test_setting](https://user-images.githubusercontent.com/50568142/139358656-a96e7546-9260-41de-91a3-a5605d53c55a.png)
