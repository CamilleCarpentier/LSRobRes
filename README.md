# Code for the manuscript "Reinterpreting the relationship between number of species and number of links connects community structure and stability"

The file `main.py` contains a guide to use the code. 
This code is designed such that any network could be analysed, as long as it is represented by its adjacency matrix (an example is given in `main.py`)

The `main.py` file is divided into three parts, allowing to reproduce the results of the three main sections of the manuscript:

* **Network specific L~S relationship**

	* In-silico extinction experiments (decompositions)
	* Prediction of the *L~S* relationship
  
  *The functions used in this section are defined in the `decomposition.py` file.*


* **Predicting robustness**

	* The average number of extinctions occurring after the removal of one species
	* The relationship between the number of species lost and the number of species removed
	* Observed robustness (mean and variance) at threshold x
	* Predicted robustness at threshold x

  *The functions used in this section are defined in the `robustness.py` file.*

* **Local stability - Robustness trade-off**

	* Observed local stability (mean and variance)
	* Predicted local stability
  
  *The functions used in this section are defined in the `local_stability.py` file.*

Written in Python 3.7.6 using 3 libraries: numpy 1.18.1, matplotlib 3.1.3 and collections.

For any question or comment, please contact: Camille Carpentier via camille.carpentier@unamur.be 
