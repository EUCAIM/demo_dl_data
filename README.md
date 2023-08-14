# EUCAIM - FL DL DEMO DATA AND MODEL

## Data description

### Datasource

The data comes from the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) data-set(s) in publicly available from [kaggle](https://www.kaggle.com)

### Preparation

The data was prepared as seen in the following R code:

```r

# FULL LIST OF FILES - TRAIN
trn.nrm.all <- list.files( "~/Projects/eucaim_dl_model/chest_xray/train/NORMAL/" )
trn.pnm.all <- list.files( "~/Projects/eucaim_dl_model/chest_xray/train/PNEUMONIA/" )

# FULL LIST OF FILES - TEST
tst.nrm.all <- list.files( "~/Projects/eucaim_dl_model/chest_xray/test/NORMAL/" )
tst.pnm.all <- list.files( "~/Projects/eucaim_dl_model/chest_xray/test/PNEUMONIA/" )

# SPLIT IN THREE RANDOM SETS
split <- function( files, ngroups = 3 ) {
  n <- length( files ) / ngroups
  files <- sample( files )
  lapply( seq( ngroups ), function( ii ) {
    start <- 1 + ( ii - 1 ) * n
    end   <- n * ii 
    if( ii == ngroups & n - floor( n ) != 0 ) {
      end <- n * ii + 1
    }
    return( files[ start: end ] )
  } )
}

trn.nrm.3 <- split( trn.nrm.all, 3 )
trn.pnm.3 <- split( trn.pnm.all, 3 )

tst.nrm.3 <- split( tst.nrm.all, 3 )
tst.pnm.3 <- split( tst.pnm.all, 3 )


save_ids <- function( split_files, filename ) {
  for( ii in seq( length( split_files ) ) ) {
    localname <- paste0( filename, ii, ".csv" )
    data <- data.frame( image_name = split_files[[ ii ]] )
    write.csv( data, file = localname, quote = FALSE, row.names = FALSE )
  }
}

save_ids( trn.nrm.3, "./data_ids/three_dataseties_scenario/train.nrm.3_" )
save_ids( trn.pnm.3, "./data_ids/three_dataseties_scenario/train.pnm.3_" )

save_ids( tst.nrm.3, "./data_ids/three_dataseties_scenario/test.nrm.3_" )
save_ids( tst.pnm.3, "./data_ids/three_dataseties_scenario/test.pnm.3_" )

# SPLINT IN TWO RANDOM SETS
trn.nrm.2 <- split( trn.nrm.all, 2 )
trn.pnm.2 <- split( trn.pnm.all, 2 )

tst.nrm.2 <- split( tst.nrm.all, 2 )
tst.pnm.2 <- split( tst.pnm.all, 2 )

save_ids( trn.nrm.2, "./data_ids/three_dataseties_scenario/train.nrm.2_" )
save_ids( trn.pnm.2, "./data_ids/three_dataseties_scenario/train.pnm.2_" )

save_ids( tst.nrm.2, "./data_ids/three_dataseties_scenario/test.nrm.2_" )
save_ids( tst.pnm.2, "./data_ids/three_dataseties_scenario/test.pnm.2_" )
```

Therefore:

 * There are two possible scenarios: two or three data-sites available
 * Each file contains a third or a half of the full data-sets
 * Each files contains the same variable (aka. `image_name`) with the name of the image that is included in the set
 
## Model description

The model to test is the following:

 * 2d Convolution layer (input layer; in: 3, out: 12)
 * 2d normalize (1st hidden layer; in: 12, out: 12)
 * Rectified linear unit (2nd hidden layer; in: 12, out: 12)
 * 2d pooling (max) (3th hidden layer; in: 12; out: 12)
 * 2d Convolution layer (4th hidden layer; in: 12, out: 20)
 * Rectified linear unit (5th hidden layer; in: 20, out: 20)
 * 2d Convolution layer (6th hidden layer; in: 20, out: 32)
 * 2d normalize (7th hidden layer; in: 32, out: 32)
 * Rectified linear unit (8th hidden layer; in: 32, out: 32)
 * Linear discriminant (output layer; in: 32, out: 2)        

Depicted as:

![CNN model drawing](https://i.ibb.co/QF0WLFm/nn.jpg)

### Ground truth

In order to validate the results of the FL DL exercise, we will take the following as the ground truth model:

```python
class CnnModel( nn.Module ) :
    def __init__( self, num_classes = 2 ):
        super( CnnModel, self ).__init__()
        
        self.conv1 = nn.Conv2d( in_channels = 3, out_channels = 12, kernel_size = 3, stride = 1, padding = 1 )
        self.bn1   = nn.BatchNorm2d( num_features = 12 )
        self.relu1 = nn.ReLU()   
        self.pool  = nn.MaxPool2d( kernel_size = 2 )
        self.conv2 = nn.Conv2d( in_channels = 12, out_channels = 20, kernel_size = 3, stride = 1, padding = 1 )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d( in_channels = 20, out_channels = 32, kernel_size = 3, stride = 1, padding = 1 )
        self.bn3   = nn.BatchNorm2d( num_features = 32 )
        self.relu3 = nn.ReLU()
        self.fc    = nn.Linear( in_features = 32 * 112 * 112, out_features = num_classes )
        
    def forward( self, input ):
        output = self.conv1( input )
        output = self.bn1( output )
        output = self.relu1( output )
        output = self.pool( output )
        output = self.conv2( output )
        output = self.relu2( output )
        output = self.conv3( output )
        output = self.bn3( output )
        output = self.relu3( output )            
        output = output.view( -1, 32*112*112 )
        output = self.fc( output )
        return output
```

The model was trained in 10 epochs and using the Adam optimizer and loss of cross entropy criterion using only local CPU, and using the full set of train data-set as seen as:

```txt
Epoch 0 - Loss: 107.02728892862797
Epoch 1 - Loss: 16.680565030500293
Epoch 2 - Loss: 7.684169912338257
Epoch 3 - Loss: 8.067254842817784
Epoch 4 - Loss: 3.4069480776786802
Epoch 5 - Loss: 4.979365282598883
Epoch 6 - Loss: 3.159924500063062
Epoch 7 - Loss: 1.560024076141417
Epoch 8 - Loss: 1.0648583032190797
Epoch 9 - Loss: 0.7512097196653486
Model test accuracy: 0.7756410256410257; Model test loss: 19.418610334396362
```
