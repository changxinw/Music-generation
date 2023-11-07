# Conditional Deep Music Generation Using Modified VAE

The deep music generation models allow conditional generation over different pianists and fuses the styles of the composers.

## Directory Introduction
### Jupyter Notebooks

- **Dataprocessing.ipynb**: To implement the data processing of midi files, including read from directory, transform files into one hot vectors and cut the notes into fixed length to train.

- **CVAE.ipynb**: To bulid conditional VAE model

- **Right and left Rolls.ipynb**: To bulid modified MIDI-VAE with right piano and left piano rolls as input

- **Three Rolls.ipynb**: To build modified MIDI-VAE with pitch, duration and velocity rolls.

- **Pitch and duration rolls.ipynb**: To build modified MIDI-VAE with pitch and duration rolls.


### Other files

- **Data.rar**: Training midi files of three different composers. The original kaggle database can be referred [here](https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi).

- **music_cvae.mid(.mp3)**: Generated MIDI music file using CVAE model with right piano and left piano rolls as input

- **music_right_left.mid(.mp3)**: Generated MIDI music file using model with right piano and left piano rolls as input

- **music_3rolls.mid(.mp3)**: Generated MIDI music file using model with pitch, duration and velocity rolls as input

- **music_2rolls.mid(.mp3)**: Generated MIDI music file using model with pitch and duration rolls as input

Reader may unzip the Data.rar and run Dataprocessing.ipynb notebook to generate the training data. Then choose any model to train and generate conditional music. 


## Model Constructure
### Conditional Variational Auto-Encoders (CVAE)

![CVAE_NEW](https://user-images.githubusercontent.com/76429734/204684867-f96e6b5d-e83b-4bc7-9844-88ad89cbb214.png)
We first implemented CVAE to generate mixed-style music. Considering the piano pieces have two tracks, the left piano and right piano, we design our
encoder and decoder to be two parallel LSTMs. After concatenating the piano's composer labels (one-hot encoding) to the piano’s notes information, and passing through the encoder, we get the latent space conditioned on the input’s composerlabels. Then we concatenate composer labels to the latent space and feed them into the decoder to get the reconstructed input.

After finishing the training stage, in the generation phase, we feed different possibilities times composers’ one-hot encoding and the latent space into the decoder and get our creative new music with mixed styles

### Modified latent space of MIDI-VAE

The models are all modified MIDI-VAE. The original MIDI-VAE can be referred [here](https://arxiv.org/pdf/1809.07600.pdf).


![network (1)](https://user-images.githubusercontent.com/97444802/204424010-0179cf97-1c22-4f31-a1ad-b6620a6e4619.png)

As Figure shown above,  we include a composer one hot vector to compute the latent variables. Same color means the representation of the same composer. A distribution including $\mu$ and $\sigma$ is expected to be learned for each composer during training. To be specific, we define $\mu = [\mu_1, \mu_2, \mu_3] $, $\sigma = [\sigma_1, \sigma_2, \sigma_3]$, composer one hot= $[\alpha_1, \alpha_2, \alpha_3] $, then the latent variable is computed using a mixture gaussian model as 

$$ z =  \sum_{i=1}^3 \alpha_i \mathcal{N}(\mu_i, \sigma_i) $$ 

During training, we use one hot vector, such as $[1,0,0]$, to only consider the distribution of a single composer. Then we can use a vector representing the probabilities of three composers that sums to 1 when generating short music. Same equation to calculate the latent variables, and we can mix different composer styles in this way. 

Three different input structures are attempted with this network structure. The figure shows the right hand piano roll and left hand piano roll as input. We also implemented structures only using right hand piano part, but dividing into pitch, duration and velocity rolls. 
