# This is a sample main file highlighting the usage of DCGAN module in DCGAN.py
# Please edit this file based on your requirements
import sys
import DCGAN as dc
import data_utilities as d_u

# Dataset
dset	= sys.argv[1]
root	= sys.argv[2]

# Arguments passed to dataset loaders can be modified based on required usage. Please look at the documentation of the loader functions using
# help(d_u) command.

if dset == 'mnist':
	dataset	= d_u.MNIST_loader(root=root, image_size=32)
	n_chan	= 1
elif dset == 'cifar10':
	dataset	= d_u.CIFAR10_loader(root=root, image_size=32, normalize=True)
	n_chan	= 3
elif dset == 'lsun':
	dataset	= d_u.LSUN_loader(root=root, image_size=64, classes=['bedroom_train'], normalize=True)
	n_chan	= 3
elif dset == 'cub':
	dataset	= d_u.CUB2011_loader(root=root, image_size=64, normalize=True)
	n_chan	= 3

# DCGAN object initialization
# Parameters below can be modified
# Please check the documentation using help(dc.DCGAN.__init__)
image_size	= 64
n_z		= 100#128
hiddens		= {'gen':	128,
		   'dis':	128
		  }
ngpu		= 1
loss		= 'BCE'

Gen_model	= dc.DCGAN(image_size=image_size, n_z=n_z, n_chan=n_chan, hiddens=hiddens, ngpu=ngpu, loss=loss)

# DCGAN training scheme
# Parameters below can be modified based on required usage.
# Please check the documentation using help(dc.DCGAN.train) for more details

batch_size	= 128
n_iters		= 7031250#1e05
opt_dets	= {'gen':	{'name'		: 'adam',
				 'learn_rate'	: 0.0002, #1e-04
				 'betas'	: (0.5, 0.99)
				},
		   'dis':	{'name'		: 'adam',
		   		 'learn_rate'	: 0.0002,
		   		 'betas'	: (0.5, 0.99)
		   		}
		  }

# Optional arguments
show_period	= 1000  # dump generated images and models per 1,000 iters = 1,000 minibatches
display_images	= True
misc_options	= ['init_scheme', 'save_model']

# Call training
Gen_model.train(dataset=dataset, batch_size=batch_size, n_iters=n_iters, optimizer_details=opt_dets, show_period=show_period, display_images=display_images, misc_options=misc_options)

# Voila, your work is done
