# Code for the manuscript "A new link-species relationship connects ecosystem structure and stability"

The file `main.py` contains a guide to use the code. 
This code is designed such that any network could be analysed, as long as it is represented by its adjacency matrix (an example is given in `main.py`)

The `main.py` file is divided into three parts, allowing to reproduce the results of the three main sections of the manuscript:

* **A new, network-specific approach**

	* In-silico extinction experiments (decompositions)
	* Prediction of the *L~S* relationship
  
  *The functions used in this section are defined in the `decomposition.py` file.*


* **Predicting robustness**

	* The average number of extinctions occurring after the removal of one species
	* The relationship between the number of species lost and the number of species removed
	* Observed robustness (mean and variance) at threshold x
	* Predicted robustness at threshold x

  *The functions used in this section are defined in the `robustness.py` file.*

* **A Resilience - Robustness trade-off**

	* Observed resilience (mean and variance)
	* Predicted resilience
  
  *The functions used in this section are defined in the `resilience.py` file.*


For any question or comment, please contact: Camille Carpentier via camille.carpentier@unamur.be 
