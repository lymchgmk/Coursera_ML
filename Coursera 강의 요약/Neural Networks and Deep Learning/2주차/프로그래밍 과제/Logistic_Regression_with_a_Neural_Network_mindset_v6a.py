{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Logistic Regression with a Neural Network mindset\n",
    "\n",
    "Welcome to your first (required) programming assignment! You will build a logistic regression classifier to recognize  cats. This assignment will step you through how to do this with a Neural Network mindset, and so will also hone your intuitions about deep learning.\n",
    "\n",
    "**Instructions:**\n",
    "- Do not use loops (for/while) in your code, unless the instructions explicitly ask you to do so.\n",
    "\n",
    "**You will learn to:**\n",
    "- Build the general architecture of a learning algorithm, including:\n",
    "    - Initializing parameters\n",
    "    - Calculating the cost function and its gradient\n",
    "    - Using an optimization algorithm (gradient descent) \n",
    "- Gather all three functions above into a main model function, in the right order."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## <font color='darkblue'>Updates</font>\n",
    "This notebook has been updated over the past few months.  The prior version was named \"v5\", and the current versionis now named '6a'\n",
    "\n",
    "#### If you were working on a previous version:\n",
    "* You can find your prior work by looking in the file directory for the older files (named by version name).\n",
    "* To view the file directory, click on the \"Coursera\" icon in the top left corner of this notebook.\n",
    "* Please copy your work from the older versions to the new version, in order to submit your work for grading.\n",
    "\n",
    "#### List of Updates\n",
    "* Forward propagation formula, indexing now starts at 1 instead of 0.\n",
    "* Optimization function comment now says \"print cost every 100 training iterations\" instead of \"examples\".\n",
    "* Fixed grammar in the comments.\n",
    "* Y_prediction_test variable name is used consistently.\n",
    "* Plot's axis label now says \"iterations (hundred)\" instead of \"iterations\".\n",
    "* When testing the model, the test image is normalized by dividing by 255."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1 - Packages ##\n",
    "\n",
    "First, let's run the cell below to import all the packages that you will need during this assignment. \n",
    "- [numpy](www.numpy.org) is the fundamental package for scientific computing with Python.\n",
    "- [h5py](http://www.h5py.org) is a common package to interact with a dataset that is stored on an H5 file.\n",
    "- [matplotlib](http://matplotlib.org) is a famous library to plot graphs in Python.\n",
    "- [PIL](http://www.pythonware.com/products/pil/) and [scipy](https://www.scipy.org/) are used here to test your model with your own picture at the end."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import h5py\n",
    "import scipy\n",
    "from PIL import Image\n",
    "from scipy import ndimage\n",
    "from lr_utils import load_dataset\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2 - Overview of the Problem set ##\n",
    "\n",
    "**Problem Statement**: You are given a dataset (\"data.h5\") containing:\n",
    "    - a training set of m_train images labeled as cat (y=1) or non-cat (y=0)\n",
    "    - a test set of m_test images labeled as cat or non-cat\n",
    "    - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px).\n",
    "\n",
    "You will build a simple image-recognition algorithm that can correctly classify pictures as cat or non-cat.\n",
    "\n",
    "Let's get more familiar with the dataset. Load the data by running the following code."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Loading the data (cat/non-cat)\n",
    "train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We added \"_orig\" at the end of image datasets (train and test) because we are going to preprocess them. After preprocessing, we will end up with train_set_x and test_set_x (the labels train_set_y and test_set_y don't need any preprocessing).\n",
    "\n",
    "Each line of your train_set_x_orig and test_set_x_orig is an array representing an image. You can visualize an example by running the following code. Feel free also to change the `index` value and re-run to see other images. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "y = [1], it's a 'cat' picture.\n"
     ]
    },
    {
     "data": {
      "image/png": "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\nK+0AAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f7ce49ef748>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Example of a picture\n",
    "index = 25\n",
    "plt.imshow(train_set_x_orig[index])\n",
    "print (\"y = \" + str(train_set_y[:, index]) + \", it's a '\" + classes[np.squeeze(train_set_y[:, index])].decode(\"utf-8\") +  \"' picture.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Many software bugs in deep learning come from having matrix/vector dimensions that don't fit. If you can keep your matrix/vector dimensions straight you will go a long way toward eliminating many bugs. \n",
    "\n",
    "**Exercise:** Find the values for:\n",
    "    - m_train (number of training examples)\n",
    "    - m_test (number of test examples)\n",
    "    - num_px (= height = width of a training image)\n",
    "Remember that `train_set_x_orig` is a numpy-array of shape (m_train, num_px, num_px, 3). For instance, you can access `m_train` by writing `train_set_x_orig.shape[0]`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of training examples: m_train = 209\n",
      "Number of testing examples: m_test = 50\n",
      "Height/Width of each image: num_px = 64\n",
      "Each image is of size: (64, 64, 3)\n",
      "train_set_x shape: (209, 64, 64, 3)\n",
      "train_set_y shape: (1, 209)\n",
      "test_set_x shape: (50, 64, 64, 3)\n",
      "test_set_y shape: (1, 50)\n"
     ]
    }
   ],
   "source": [
    "### START CODE HERE ### (≈ 3 lines of code)\n",
    "m_train = train_set_x_orig.shape[0]\n",
    "m_test = test_set_x_orig.shape[0]\n",
    "num_px = train_set_x_orig.shape[1]\n",
    "### END CODE HERE ###\n",
    "\n",
    "print (\"Number of training examples: m_train = \" + str(m_train))\n",
    "print (\"Number of testing examples: m_test = \" + str(m_test))\n",
    "print (\"Height/Width of each image: num_px = \" + str(num_px))\n",
    "print (\"Each image is of size: (\" + str(num_px) + \", \" + str(num_px) + \", 3)\")\n",
    "print (\"train_set_x shape: \" + str(train_set_x_orig.shape))\n",
    "print (\"train_set_y shape: \" + str(train_set_y.shape))\n",
    "print (\"test_set_x shape: \" + str(test_set_x_orig.shape))\n",
    "print (\"test_set_y shape: \" + str(test_set_y.shape))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Expected Output for m_train, m_test and num_px**: \n",
    "<table style=\"width:15%\">\n",
    "  <tr>\n",
    "    <td>**m_train**</td>\n",
    "    <td> 209 </td> \n",
    "  </tr>\n",
    "  \n",
    "  <tr>\n",
    "    <td>**m_test**</td>\n",
    "    <td> 50 </td> \n",
    "  </tr>\n",
    "  \n",
    "  <tr>\n",
    "    <td>**num_px**</td>\n",
    "    <td> 64 </td> \n",
    "  </tr>\n",
    "  \n",
    "</table>\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For convenience, you should now reshape images of shape (num_px, num_px, 3) in a numpy-array of shape (num_px $*$ num_px $*$ 3, 1). After this, our training (and test) dataset is a numpy-array where each column represents a flattened image. There should be m_train (respectively m_test) columns.\n",
    "\n",
    "**Exercise:** Reshape the training and test data sets so that images of size (num_px, num_px, 3) are flattened into single vectors of shape (num\\_px $*$ num\\_px $*$ 3, 1).\n",
    "\n",
    "A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b$*$c$*$d, a) is to use: \n",
    "```python\n",
    "X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train_set_x_flatten shape: (12288, 209)\n",
      "train_set_y shape: (1, 209)\n",
      "test_set_x_flatten shape: (12288, 50)\n",
      "test_set_y shape: (1, 50)\n",
      "sanity check after reshaping: [17 31 56 22 33]\n"
     ]
    }
   ],
   "source": [
    "# Reshape the training and test examples\n",
    "\n",
    "### START CODE HERE ### (≈ 2 lines of code)\n",
    "train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T\n",
    "test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T\n",
    "### END CODE HERE ###\n",
    "\n",
    "print (\"train_set_x_flatten shape: \" + str(train_set_x_flatten.shape))\n",
    "print (\"train_set_y shape: \" + str(train_set_y.shape))\n",
    "print (\"test_set_x_flatten shape: \" + str(test_set_x_flatten.shape))\n",
    "print (\"test_set_y shape: \" + str(test_set_y.shape))\n",
    "print (\"sanity check after reshaping: \" + str(train_set_x_flatten[0:5,0]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Expected Output**: \n",
    "\n",
    "<table style=\"width:35%\">\n",
    "  <tr>\n",
    "    <td>**train_set_x_flatten shape**</td>\n",
    "    <td> (12288, 209)</td> \n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>**train_set_y shape**</td>\n",
    "    <td>(1, 209)</td> \n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>**test_set_x_flatten shape**</td>\n",
    "    <td>(12288, 50)</td> \n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td>**test_set_y shape**</td>\n",
    "    <td>(1, 50)</td> \n",
    "  </tr>\n",
    "  <tr>\n",
    "  <td>**sanity check after reshaping**</td>\n",
    "  <td>[17 31 56 22 33]</td> \n",
    "  </tr>\n",
    "</table>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To represent color images, the red, green and blue channels (RGB) must be specified for each pixel, and so the pixel value is actually a vector of three numbers ranging from 0 to 255.\n",
    "\n",
    "One common preprocessing step in machine learning is to center and standardize your dataset, meaning that you substract the mean of the whole numpy array from each example, and then divide each example by the standard deviation of the whole numpy array. But for picture datasets, it is simpler and more convenient and works almost as well to just divide every row of the dataset by 255 (the maximum value of a pixel channel).\n",
    "\n",
    "<!-- During the training of your model, you're going to multiply weights and add biases to some initial inputs in order to observe neuron activations. Then you backpropogate with the gradients to train the model. But, it is extremely important for each feature to have a similar range such that our gradients don't explode. You will see that more in detail later in the lectures. !--> \n",
    "\n",
    "Let's standardize our dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_set_x = train_set_x_flatten/255.\n",
    "test_set_x = test_set_x_flatten/255."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<font color='blue'>\n",
    "**What you need to remember:**\n",
    "\n",
    "Common steps for pre-processing a new dataset are:\n",
    "- Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)\n",
    "- Reshape the datasets such that each example is now a vector of size (num_px \\* num_px \\* 3, 1)\n",
    "- \"Standardize\" the data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3 - General Architecture of the learning algorithm ##\n",
    "\n",
    "It's time to design a simple algorithm to distinguish cat images from non-cat images.\n",
    "\n",
    "You will build a Logistic Regression, using a Neural Network mindset. The following Figure explains why **Logistic Regression is actually a very simple Neural Network!**\n",
    "\n",
    "<img src=\"images/LogReg_kiank.png\" style=\"width:650px;height:400px;\">\n",
    "\n",
    "**Mathematical expression of the algorithm**:\n",
    "\n",
    "For one example $x^{(i)}$:\n",
    "$$z^{(i)} = w^T x^{(i)} + b \\tag{1}$$\n",
    "$$\\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})\\tag{2}$$ \n",
    "$$ \\mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \\log(a^{(i)}) - (1-y^{(i)} )  \\log(1-a^{(i)})\\tag{3}$$\n",
    "\n",
    "The cost is then computed by summing over all training examples:\n",
    "$$ J = \\frac{1}{m} \\sum_{i=1}^m \\mathcal{L}(a^{(i)}, y^{(i)})\\tag{6}$$\n",
    "\n",
    "**Key steps**:\n",
    "In this exercise, you will carry out the following steps: \n",
    "    - Initialize the parameters of the model\n",
    "    - Learn the parameters for the model by minimizing the cost  \n",
    "    - Use the learned parameters to make predictions (on the test set)\n",
    "    - Analyse the results and conclude"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4 - Building the parts of our algorithm ## \n",
    "\n",
    "The main steps for building a Neural Network are:\n",
    "1. Define the model structure (such as number of input features) \n",
    "2. Initialize the model's parameters\n",
    "3. Loop:\n",
    "    - Calculate current loss (forward propagation)\n",
    "    - Calculate current gradient (backward propagation)\n",
    "    - Update parameters (gradient descent)\n",
    "\n",
    "You often build 1-3 separately and integrate them into one function we call `model()`.\n",
    "\n",
    "### 4.1 - Helper functions\n",
    "\n",
    "**Exercise**: Using your code from \"Python Basics\", implement `sigmoid()`. As you've seen in the figure above, you need to compute $sigmoid( w^T x + b) = \\frac{1}{1 + e^{-(w^T x + b)}}$ to make predictions. Use np.exp()."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "# GRADED FUNCTION: sigmoid\n",
    "\n",
    "def sigmoid(z):\n",
    "    \"\"\"\n",
    "    Compute the sigmoid of z\n",
    "\n",
    "    Arguments:\n",
    "    z -- A scalar or numpy array of any size.\n",
    "\n",
    "    Return:\n",
    "    s -- sigmoid(z)\n",
    "    \"\"\"\n",
    "\n",
    "    ### START CODE HERE ### (≈ 1 line of code)\n",
    "    s = 1/(1+np.exp(-z))\n",
    "    ### END CODE HERE ###\n",
    "    \n",
    "    return s"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "sigmoid([0, 2]) = [ 0.5         0.88079708]\n"
     ]
    }
   ],
   "source": [
    "print (\"sigmoid([0, 2]) = \" + str(sigmoid(np.array([0,2]))))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Expected Output**: \n",
    "\n",
    "<table>\n",
    "  <tr>\n",
    "    <td>**sigmoid([0, 2])**</td>\n",
    "    <td> [ 0.5         0.88079708]</td> \n",
    "  </tr>\n",
    "</table>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.2 - Initializing parameters\n",
    "\n",
    "**Exercise:** Implement parameter initialization in the cell below. You have to initialize w as a vector of zeros. If you don't know what numpy function to use, look up np.zeros() in the Numpy library's documentation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "# GRADED FUNCTION: initialize_with_zeros\n",
    "\n",
    "def initialize_with_zeros(dim):\n",
    "    \"\"\"\n",
    "    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.\n",
    "    \n",
    "    Argument:\n",
    "    dim -- size of the w vector we want (or number of parameters in this case)\n",
    "    \n",
    "    Returns:\n",
    "    w -- initialized vector of shape (dim, 1)\n",
    "    b -- initialized scalar (corresponds to the bias)\n",
    "    \"\"\"\n",
    "    \n",
    "    ### START CODE HERE ### (≈ 1 line of code)\n",
    "    w = np.zeros((dim, 1))\n",
    "    b = 0\n",
    "    ### END CODE HERE ###\n",
    "\n",
    "    assert(w.shape == (dim, 1))\n",
    "    assert(isinstance(b, float) or isinstance(b, int))\n",
    "    \n",
    "    return w, b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "w = [[ 0.]\n",
      " [ 0.]]\n",
      "b = 0\n"
     ]
    }
   ],
   "source": [
    "dim = 2\n",
    "w, b = initialize_with_zeros(dim)\n",
    "print (\"w = \" + str(w))\n",
    "print (\"b = \" + str(b))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Expected Output**: \n",
    "\n",
    "\n",
    "<table style=\"width:15%\">\n",
    "    <tr>\n",
    "        <td>  ** w **  </td>\n",
    "        <td> [[ 0.]\n",
    " [ 0.]] </td>\n",
    "    </tr>\n",
    "    <tr>\n",
    "        <td>  ** b **  </td>\n",
    "        <td> 0 </td>\n",
    "    </tr>\n",
    "</table>\n",
    "\n",
    "For image inputs, w will be of shape (num_px $\\times$ num_px $\\times$ 3, 1)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.3 - Forward and Backward propagation\n",
    "\n",
    "Now that your parameters are initialized, you can do the \"forward\" and \"backward\" propagation steps for learning the parameters.\n",
    "\n",
    "**Exercise:** Implement a function `propagate()` that computes the cost function and its gradient.\n",
    "\n",
    "**Hints**:\n",
    "\n",
    "Forward Propagation:\n",
    "- You get X\n",
    "- You compute $A = \\sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})$\n",
    "- You calculate the cost function: $J = -\\frac{1}{m}\\sum_{i=1}^{m}y^{(i)}\\log(a^{(i)})+(1-y^{(i)})\\log(1-a^{(i)})$\n",
    "\n",
    "Here are the two formulas you will be using: \n",
    "\n",
    "$$ \\frac{\\partial J}{\\partial w} = \\frac{1}{m}X(A-Y)^T\\tag{7}$$\n",
    "$$ \\frac{\\partial J}{\\partial b} = \\frac{1}{m} \\sum_{i=1}^m (a^{(i)}-y^{(i)})\\tag{8}$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# GRADED FUNCTION: propagate\n",
    "\n",
    "def propagate(w, b, X, Y):\n",
    "    \"\"\"\n",
    "    Implement the cost function and its gradient for the propagation explained above\n",
    "\n",
    "    Arguments:\n",
    "    w -- weights, a numpy array of size (num_px * num_px * 3, 1)\n",
    "    b -- bias, a scalar\n",
    "    X -- data of size (num_px * num_px * 3, number of examples)\n",
    "    Y -- true \"label\" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)\n",
    "\n",
    "    Return:\n",
    "    cost -- negative log-likelihood cost for logistic regression\n",
    "    dw -- gradient of the loss with respect to w, thus same shape as w\n",
    "    db -- gradient of the loss with respect to b, thus same shape as b\n",
    "    \n",
    "    Tips:\n",
    "    - Write your code step by step for the propagation. np.log(), np.dot()\n",
    "    \"\"\"\n",
    "    \n",
    "    m = X.shape[1]\n",
    "    \n",
    "    # FORWARD PROPAGATION (FROM X TO COST)\n",
    "    ### START CODE HERE ### (≈ 2 lines of code)\n",
    "    A = sigmoid(np.dot(w.T, X) + b)                                    # compute activation\n",
    "    cost = (-1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))                                 # compute cost\n",
    "    ### END CODE HERE ###\n",
    "    \n",
    "    # BACKWARD PROPAGATION (TO FIND GRAD)\n",
    "    ### START CODE HERE ### (≈ 2 lines of code)\n",
    "    dw = (1/m)*np.dot(X, (A-Y).T)\n",
    "    db = (1/m)*np.sum(A-Y)\n",
    "    ### END CODE HERE ###\n",
    "\n",
    "    assert(dw.shape == w.shape)\n",
    "    assert(db.dtype == float)\n",
    "    cost = np.squeeze(cost)\n",
    "    assert(cost.shape == ())\n",
    "    \n",
    "    grads = {\"dw\": dw,\n",
    "             \"db\": db}\n",
    "    \n",
    "    return grads, cost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "dw = [[ 0.99845601]\n",
      " [ 2.39507239]]\n",
      "db = 0.00145557813678\n",
      "cost = 5.80154531939\n"
     ]
    }
   ],
   "source": [
    "w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])\n",
    "grads, cost = propagate(w, b, X, Y)\n",
    "print (\"dw = \" + str(grads[\"dw\"]))\n",
    "print (\"db = \" + str(grads[\"db\"]))\n",
    "print (\"cost = \" + str(cost))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Expected Output**:\n",
    "\n",
    "<table style=\"width:50%\">\n",
    "    <tr>\n",
    "        <td>  ** dw **  </td>\n",
    "      <td> [[ 0.99845601]\n",
    "     [ 2.39507239]]</td>\n",
    "    </tr>\n",
    "    <tr>\n",
    "        <td>  ** db **  </td>\n",
    "        <td> 0.00145557813678 </td>\n",
    "    </tr>\n",
    "    <tr>\n",
    "        <td>  ** cost **  </td>\n",
    "        <td> 5.801545319394553 </td>\n",
    "    </tr>\n",
    "\n",
    "</table>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.4 - Optimization\n",
    "- You have initialized your parameters.\n",
    "- You are also able to compute a cost function and its gradient.\n",
    "- Now, you want to update the parameters using gradient descent.\n",
    "\n",
    "**Exercise:** Write down the optimization function. The goal is to learn $w$ and $b$ by minimizing the cost function $J$. For a parameter $\\theta$, the update rule is $ \\theta = \\theta - \\alpha \\text{ } d\\theta$, where $\\alpha$ is the learning rate."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# GRADED FUNCTION: optimize\n",
    "\n",
    "def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):\n",
    "    \"\"\"\n",
    "    This function optimizes w and b by running a gradient descent algorithm\n",
    "    \n",
    "    Arguments:\n",
    "    w -- weights, a numpy array of size (num_px * num_px * 3, 1)\n",
    "    b -- bias, a scalar\n",
    "    X -- data of shape (num_px * num_px * 3, number of examples)\n",
    "    Y -- true \"label\" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)\n",
    "    num_iterations -- number of iterations of the optimization loop\n",
    "    learning_rate -- learning rate of the gradient descent update rule\n",
    "    print_cost -- True to print the loss every 100 steps\n",
    "    \n",
    "    Returns:\n",
    "    params -- dictionary containing the weights w and bias b\n",
    "    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function\n",
    "    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.\n",
    "    \n",
    "    Tips:\n",
    "    You basically need to write down two steps and iterate through them:\n",
    "        1) Calculate the cost and the gradient for the current parameters. Use propagate().\n",
    "        2) Update the parameters using gradient descent rule for w and b.\n",
    "    \"\"\"\n",
    "    \n",
    "    costs = []\n",
    "    \n",
    "    for i in range(num_iterations):\n",
    "        \n",
    "        \n",
    "        # Cost and gradient calculation (≈ 1-4 lines of code)\n",
    "        ### START CODE HERE ### \n",
    "        m = X.shape[1]\n",
    "        A = sigmoid(np.dot(w.T, X) + b)\n",
    "        grads = {\"dw\": (1/m)*np.dot(X, (A-Y).T), \"db\":(1/m)*np.sum(A-Y)}\n",
    "        cost = (-1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))\n",
    "        ### END CODE HERE ###\n",
    "        \n",
    "        # Retrieve derivatives from grads\n",
    "        dw = grads[\"dw\"]\n",
    "        db = grads[\"db\"]\n",
    "        \n",
    "        # update rule (≈ 2 lines of code)\n",
    "        ### START CODE HERE ###\n",
    "        w = w - learning_rate * dw\n",
    "        b = b - learning_rate * db\n",
    "        ### END CODE HERE ###\n",
    "        \n",
    "        # Record the costs\n",
    "        if i % 100 == 0:\n",
    "            costs.append(cost)\n",
    "        \n",
    "        # Print the cost every 100 training iterations\n",
    "        if print_cost and i % 100 == 0:\n",
    "            print (\"Cost after iteration %i: %f\" %(i, cost))\n",
    "    \n",
    "    params = {\"w\": w,\n",
    "              \"b\": b}\n",
    "    \n",
    "    grads = {\"dw\": dw,\n",
    "             \"db\": db}\n",
    "    \n",
    "    return params, grads, costs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "w = [[ 0.19033591]\n",
      " [ 0.12259159]]\n",
      "b = 1.92535983008\n",
      "dw = [[ 0.67752042]\n",
      " [ 1.41625495]]\n",
      "db = 0.219194504541\n"
     ]
    }
   ],
   "source": [
    "params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)\n",
    "\n",
    "print (\"w = \" + str(params[\"w\"]))\n",
    "print (\"b = \" + str(params[\"b\"]))\n",
    "print (\"dw = \" + str(grads[\"dw\"]))\n",
    "print (\"db = \" + str(grads[\"db\"]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Expected Output**: \n",
    "\n",
    "<table style=\"width:40%\">\n",
    "    <tr>\n",
    "       <td> **w** </td>\n",
    "       <td>[[ 0.19033591]\n",
    " [ 0.12259159]] </td>\n",
    "    </tr>\n",
    "    \n",
    "    <tr>\n",
    "       <td> **b** </td>\n",
    "       <td> 1.92535983008 </td>\n",
    "    </tr>\n",
    "    <tr>\n",
    "       <td> **dw** </td>\n",
    "       <td> [[ 0.67752042]\n",
    " [ 1.41625495]] </td>\n",
    "    </tr>\n",
    "    <tr>\n",
    "       <td> **db** </td>\n",
    "       <td> 0.219194504541 </td>\n",
    "    </tr>\n",
    "\n",
    "</table>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Exercise:** The previous function will output the learned w and b. We are able to use w and b to predict the labels for a dataset X. Implement the `predict()` function. There are two steps to computing predictions:\n",
    "\n",
    "1. Calculate $\\hat{Y} = A = \\sigma(w^T X + b)$\n",
    "\n",
    "2. Convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5), stores the predictions in a vector `Y_prediction`. If you wish, you can use an `if`/`else` statement in a `for` loop (though there is also a way to vectorize this). "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# GRADED FUNCTION: predict\n",
    "\n",
    "def predict(w, b, X):\n",
    "    '''\n",
    "    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)\n",
    "    \n",
    "    Arguments:\n",
    "    w -- weights, a numpy array of size (num_px * num_px * 3, 1)\n",
    "    b -- bias, a scalar\n",
    "    X -- data of size (num_px * num_px * 3, number of examples)\n",
    "    \n",
    "    Returns:\n",
    "    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X\n",
    "    '''\n",
    "    \n",
    "    m = X.shape[1]\n",
    "    Y_prediction = np.zeros((1,m))\n",
    "    w = w.reshape(X.shape[0], 1)\n",
    "    \n",
    "    # Compute vector \"A\" predicting the probabilities of a cat being present in the picture\n",
    "    ### START CODE HERE ### (≈ 1 line of code)\n",
    "    A = sigmoid(np.dot(w.T, X) + b)\n",
    "    ### END CODE HERE ###\n",
    "    for i in range(A.shape[1]):\n",
    "        # Convert probabilities A[0,i] to actual predictions p[0,i]\n",
    "        ### START CODE HERE ### (≈ 4 lines of code)\n",
    "        if A[0][i] > 0.5:\n",
    "            Y_prediction[0][i] = 1\n",
    "        else:\n",
    "            Y_prediction[0][i] = 0\n",
    "        ### END CODE HERE ###\n",
    "    \n",
    "    assert(Y_prediction.shape == (1, m))\n",
    "    \n",
    "    return Y_prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "predictions = [[ 1.  1.  0.]]\n"
     ]
    }
   ],
   "source": [
    "w = np.array([[0.1124579],[0.23106775]])\n",
    "b = -0.3\n",
    "X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])\n",
    "print (\"predictions = \" + str(predict(w, b, X)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Expected Output**: \n",
    "\n",
    "<table style=\"width:30%\">\n",
    "    <tr>\n",
    "         <td>\n",
    "             **predictions**\n",
    "         </td>\n",
    "          <td>\n",
    "            [[ 1.  1.  0.]]\n",
    "         </td>  \n",
    "   </tr>\n",
    "\n",
    "</table>\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<font color='blue'>\n",
    "**What to remember:**\n",
    "You've implemented several functions that:\n",
    "- Initialize (w,b)\n",
    "- Optimize the loss iteratively to learn parameters (w,b):\n",
    "    - computing the cost and its gradient \n",
    "    - updating the parameters using gradient descent\n",
    "- Use the learned (w,b) to predict the labels for a given set of examples"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5 - Merge all functions into a model ##\n",
    "\n",
    "You will now see how the overall model is structured by putting together all the building blocks (functions implemented in the previous parts) together, in the right order.\n",
    "\n",
    "**Exercise:** Implement the model function. Use the following notation:\n",
    "    - Y_prediction_test for your predictions on the test set\n",
    "    - Y_prediction_train for your predictions on the train set\n",
    "    - w, costs, grads for the outputs of optimize()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# GRADED FUNCTION: model\n",
    "\n",
    "def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):\n",
    "    \"\"\"\n",
    "    Builds the logistic regression model by calling the function you've implemented previously\n",
    "    \n",
    "    Arguments:\n",
    "    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)\n",
    "    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)\n",
    "    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)\n",
    "    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)\n",
    "    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters\n",
    "    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()\n",
    "    print_cost -- Set to true to print the cost every 100 iterations\n",
    "    \n",
    "    Returns:\n",
    "    d -- dictionary containing information about the model.\n",
    "    \"\"\"\n",
    "    \n",
    "    ### START CODE HERE ###\n",
    "    \n",
    "    # initialize parameters with zeros (≈ 1 line of code)\n",
    "    w, b = np.zeros((X_train.shape[0], 1)), 0\n",
    "\n",
    "    # Gradient descent (≈ 1 line of code)\n",
    "    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)\n",
    "    \n",
    "    # Retrieve parameters w and b from dictionary \"parameters\"\n",
    "    w = parameters[\"w\"]\n",
    "    b = parameters[\"b\"]\n",
    "    \n",
    "    # Predict test/train set examples (≈ 2 lines of code)\n",
    "    Y_prediction_test = predict(w, b, X_test)\n",
    "    Y_prediction_train = predict(w, b, X_train)\n",
    "\n",
    "    ### END CODE HERE ###\n",
    "\n",
    "    # Print train/test Errors\n",
    "    print(\"train accuracy: {} %\".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))\n",
    "    print(\"test accuracy: {} %\".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))\n",
    "\n",
    "    \n",
    "    d = {\"costs\": costs,\n",
    "         \"Y_prediction_test\": Y_prediction_test, \n",
    "         \"Y_prediction_train\" : Y_prediction_train, \n",
    "         \"w\" : w, \n",
    "         \"b\" : b,\n",
    "         \"learning_rate\" : learning_rate,\n",
    "         \"num_iterations\": num_iterations}\n",
    "    \n",
    "    return d"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Run the following cell to train your model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cost after iteration 0: 0.693147\n",
      "Cost after iteration 100: 0.584508\n",
      "Cost after iteration 200: 0.466949\n",
      "Cost after iteration 300: 0.376007\n",
      "Cost after iteration 400: 0.331463\n",
      "Cost after iteration 500: 0.303273\n",
      "Cost after iteration 600: 0.279880\n",
      "Cost after iteration 700: 0.260042\n",
      "Cost after iteration 800: 0.242941\n",
      "Cost after iteration 900: 0.228004\n",
      "Cost after iteration 1000: 0.214820\n",
      "Cost after iteration 1100: 0.203078\n",
      "Cost after iteration 1200: 0.192544\n",
      "Cost after iteration 1300: 0.183033\n",
      "Cost after iteration 1400: 0.174399\n",
      "Cost after iteration 1500: 0.166521\n",
      "Cost after iteration 1600: 0.159305\n",
      "Cost after iteration 1700: 0.152667\n",
      "Cost after iteration 1800: 0.146542\n",
      "Cost after iteration 1900: 0.140872\n",
      "train accuracy: 99.04306220095694 %\n",
      "test accuracy: 70.0 %\n"
     ]
    }
   ],
   "source": [
    "d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Expected Output**: \n",
    "\n",
    "<table style=\"width:40%\"> \n",
    "\n",
    "    <tr>\n",
    "        <td> **Cost after iteration 0 **  </td> \n",
    "        <td> 0.693147 </td>\n",
    "    </tr>\n",
    "      <tr>\n",
    "        <td> <center> $\\vdots$ </center> </td> \n",
    "        <td> <center> $\\vdots$ </center> </td> \n",
    "    </tr>  \n",
    "    <tr>\n",
    "        <td> **Train Accuracy**  </td> \n",
    "        <td> 99.04306220095694 % </td>\n",
    "    </tr>\n",
    "\n",
    "    <tr>\n",
    "        <td>**Test Accuracy** </td> \n",
    "        <td> 70.0 % </td>\n",
    "    </tr>\n",
    "</table> \n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Comment**: Training accuracy is close to 100%. This is a good sanity check: your model is working and has high enough capacity to fit the training data. Test accuracy is 68%. It is actually not bad for this simple model, given the small dataset we used and that logistic regression is a linear classifier. But no worries, you'll build an even better classifier next week!\n",
    "\n",
    "Also, you see that the model is clearly overfitting the training data. Later in this specialization you will learn how to reduce overfitting, for example by using regularization. Using the code below (and changing the `index` variable) you can look at predictions on pictures of the test set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "y = 1, you predicted that it is a \"cat\" picture.\n"
     ]
    },
    {
     "data": {
      "image/png": "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\nSUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f7cdf7f61d0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Example of a picture that was wrongly classified.\n",
    "index = 1\n",
    "plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))\n",
    "print (\"y = \" + str(test_set_y[0,index]) + \", you predicted that it is a \\\"\" + classes[d[\"Y_prediction_test\"][0,index]].decode(\"utf-8\") +  \"\\\" picture.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's also plot the cost function and the gradients."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f7cdc309748>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot learning curve (with costs)\n",
    "costs = np.squeeze(d['costs'])\n",
    "plt.plot(costs)\n",
    "plt.ylabel('cost')\n",
    "plt.xlabel('iterations (per hundreds)')\n",
    "plt.title(\"Learning rate =\" + str(d[\"learning_rate\"]))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Interpretation**:\n",
    "You can see the cost decreasing. It shows that the parameters are being learned. However, you see that you could train the model even more on the training set. Try to increase the number of iterations in the cell above and rerun the cells. You might see that the training set accuracy goes up, but the test set accuracy goes down. This is called overfitting. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6 - Further analysis (optional/ungraded exercise) ##\n",
    "\n",
    "Congratulations on building your first image classification model. Let's analyze it further, and examine possible choices for the learning rate $\\alpha$. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Choice of learning rate ####\n",
    "\n",
    "**Reminder**:\n",
    "In order for Gradient Descent to work you must choose the learning rate wisely. The learning rate $\\alpha$  determines how rapidly we update the parameters. If the learning rate is too large we may \"overshoot\" the optimal value. Similarly, if it is too small we will need too many iterations to converge to the best values. That's why it is crucial to use a well-tuned learning rate.\n",
    "\n",
    "Let's compare the learning curve of our model with several choices of learning rates. Run the cell below. This should take about 1 minute. Feel free also to try different values than the three we have initialized the `learning_rates` variable to contain, and see what happens. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "learning_rates = [0.01, 0.001, 0.0001]\n",
    "models = {}\n",
    "for i in learning_rates:\n",
    "    print (\"learning rate is: \" + str(i))\n",
    "    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)\n",
    "    print ('\\n' + \"-------------------------------------------------------\" + '\\n')\n",
    "\n",
    "for i in learning_rates:\n",
    "    plt.plot(np.squeeze(models[str(i)][\"costs\"]), label= str(models[str(i)][\"learning_rate\"]))\n",
    "\n",
    "plt.ylabel('cost')\n",
    "plt.xlabel('iterations (hundreds)')\n",
    "\n",
    "legend = plt.legend(loc='upper center', shadow=True)\n",
    "frame = legend.get_frame()\n",
    "frame.set_facecolor('0.90')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Interpretation**: \n",
    "- Different learning rates give different costs and thus different predictions results.\n",
    "- If the learning rate is too large (0.01), the cost may oscillate up and down. It may even diverge (though in this example, using 0.01 still eventually ends up at a good value for the cost). \n",
    "- A lower cost doesn't mean a better model. You have to check if there is possibly overfitting. It happens when the training accuracy is a lot higher than the test accuracy.\n",
    "- In deep learning, we usually recommend that you: \n",
    "    - Choose the learning rate that better minimizes the cost function.\n",
    "    - If your model overfits, use other techniques to reduce overfitting. (We'll talk about this in later videos.) \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7 - Test with your own image (optional/ungraded exercise) ##\n",
    "\n",
    "Congratulations on finishing this assignment. You can use your own image and see the output of your model. To do that:\n",
    "    1. Click on \"File\" in the upper bar of this notebook, then click \"Open\" to go on your Coursera Hub.\n",
    "    2. Add your image to this Jupyter Notebook's directory, in the \"images\" folder\n",
    "    3. Change your image's name in the following code\n",
    "    4. Run the code and check if the algorithm is right (1 = cat, 0 = non-cat)!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: 'images/lymchgmk.jpg'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-35-3a018371664f>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;31m# We preprocess the image to fit your algorithm.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0mfname\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m\"images/\"\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0mmy_image\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 7\u001b[0;31m \u001b[0mimage\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0marray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mndimage\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mimread\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfname\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mflatten\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      8\u001b[0m \u001b[0mimage\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mimage\u001b[0m\u001b[0;34m/\u001b[0m\u001b[0;36m255.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      9\u001b[0m \u001b[0mmy_image\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mscipy\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmisc\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mimresize\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mimage\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0msize\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnum_px\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mnum_px\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreshape\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnum_px\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0mnum_px\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0;36m3\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mT\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.6/site-packages/scipy/ndimage/io.py\u001b[0m in \u001b[0;36mimread\u001b[0;34m(fname, flatten, mode)\u001b[0m\n\u001b[1;32m     22\u001b[0m \u001b[0;32mdef\u001b[0m \u001b[0mimread\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfname\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mflatten\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmode\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     23\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0m_have_pil\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 24\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0m_imread\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfname\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mflatten\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmode\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     25\u001b[0m     raise ImportError(\"Could not import the Python Imaging Library (PIL)\"\n\u001b[1;32m     26\u001b[0m                       \u001b[0;34m\" required to load image files.  Please refer to\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.6/site-packages/scipy/misc/pilutil.py\u001b[0m in \u001b[0;36mimread\u001b[0;34m(name, flatten, mode)\u001b[0m\n\u001b[1;32m    152\u001b[0m     \"\"\"\n\u001b[1;32m    153\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 154\u001b[0;31m     \u001b[0mim\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mImage\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mname\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    155\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mfromimage\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mim\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mflatten\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mflatten\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmode\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mmode\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    156\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.6/site-packages/PIL/Image.py\u001b[0m in \u001b[0;36mopen\u001b[0;34m(fp, mode)\u001b[0m\n\u001b[1;32m   2310\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2311\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mfilename\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2312\u001b[0;31m         \u001b[0mfp\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mbuiltins\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilename\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"rb\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2313\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2314\u001b[0m     \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: 'images/lymchgmk.jpg'"
     ]
    }
   ],
   "source": [
    "## START CODE HERE ## (PUT YOUR IMAGE NAME) \n",
    "my_image = \"lymchgmk.jpg\"   # change this to the name of your image file \n",
    "## END CODE HERE ##\n",
    "\n",
    "# We preprocess the image to fit your algorithm.\n",
    "fname = \"images/\" + my_image\n",
    "image = np.array(ndimage.imread(fname, flatten=False))\n",
    "image = image/255.\n",
    "my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T\n",
    "my_predicted_image = predict(d[\"w\"], d[\"b\"], my_image)\n",
    "\n",
    "plt.imshow(image)\n",
    "print(\"y = \" + str(np.squeeze(my_predicted_image)) + \", your algorithm predicts a \\\"\" + classes[int(np.squeeze(my_predicted_image)),].decode(\"utf-8\") +  \"\\\" picture.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<font color='blue'>\n",
    "**What to remember from this assignment:**\n",
    "1. Preprocessing the dataset is important.\n",
    "2. You implemented each function separately: initialize(), propagate(), optimize(). Then you built a model().\n",
    "3. Tuning the learning rate (which is an example of a \"hyperparameter\") can make a big difference to the algorithm. You will see more examples of this later in this course!"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Finally, if you'd like, we invite you to try different things on this Notebook. Make sure you submit before trying anything. Once you submit, things you can play with include:\n",
    "    - Play with the learning rate and the number of iterations\n",
    "    - Try different initialization methods and compare the results\n",
    "    - Test other preprocessings (center the data, or divide each row by its standard deviation)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Bibliography:\n",
    "- http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/\n",
    "- https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c"
   ]
  }
 ],
 "metadata": {
  "coursera": {
   "course_slug": "neural-networks-deep-learning",
   "graded_item_id": "XaIWT",
   "launcher_item_id": "zAgPl"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
