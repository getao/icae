1. Our customized peft is a must to use our code to reproduce our result. You should install the peft package with the command: "pip install -e peft/"

2. icae/ contains the core code of In-context Autoencoder (ICAE) where utils/stable_trainer.py is for token-level (micro-avg) loss during training. If you don't care token-level or sample-level, you can ignore it and use the default trainer.py.  

3. transformers==4.31.0 is required. The earlier versions of transformers are not supported
