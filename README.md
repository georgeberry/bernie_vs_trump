# Bernie vs Trump classifier project

## Steps

1. Get press releases from [Bernie](http://berniesanders.com/press-release) and [Trump](http://www.donaldjtrump.com/press-releases)'s websites, store them as `.json` files.
2. Get last 3200 Tweets from [Bernie](https://twitter.com/berniesanders?lang=en) and [Trump](https://twitter.com/realDonaldTrump?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)'s Twitter accounts.
3. Get up to 3200 Tweets from a small sample of people following both Bernie and Trump.
4. Perform basic NLP feature extraction, such as constructing unigrams and bigrams, and weighting term frequencies.
5. Train a variety of classifiers (less scary than it sounds) on the Trump-Bernie Twitter data.
6. Use these classifiers to predict: 1) whether a Tweet is Trump or Bernie's; 2) Whether a Tweet comes from a Trumpist or Bernie-ite; 3) Whether a press release was from Trump or Bernie's campaign.

## Schedule

* `Lesson 1`: Step 1
* `Lesson 2`: Steps 2 and 3
* `Lesson 3`: Step 4
* `Lesson 4`: Steps 5 and 6

## Python packages we will use

### Built-ins

* json
* time
* random

### Third party

Install 3rd party packages with `pip`

For instance: `pip install python-twitter`

* python-twitter
* BeautifulSoup
* Selenium (with PhantomJS)
* Numpy/Scipy
* SciKit Learn

* PhantomJS is crucial for the crawler to work but cannot be installed via pip. The following link contains instructions necessary to install it: http://stackoverflow.com/questions/13287490/is-there-a-way-to-use-phantomjs-in-python?answertab=votes#tab-top

## Python conventions

* Put a space before a comment: `# This is a comment`
* Don't make lines longer than ~80 characters
* Constants in all-caps: `MY_CONSTANT`
* Use underscores to separate words in variable names: `my_variable`
* Avoid meaningless variable names. Avoid numbers in variable names. Wrong: `thing1`, `thing2`. Right: `cat_list`, `fluffy_cat_list`.
* When ambiguous, put variable type in name: `my_list` or `my_set`. This is particularly important for collections. Is it a `dict` or a `list`?
* Document code with triple quotes (multiline comments): `"""My documentation"""`
* Write functions when you find yourself repeating code
* When importing modules, don't import specific functions. Import the whole module, and use the module name and function together. Right: `import time; time.sleep(1)`. Wrong: `from time import sleep; sleep(1)`
* When you find yourself checking if items are in a `list`, use a `set`
* Write a snippet of documentation at the top of your file to help you remember what the file does.
* Write inputs and outputs to functions in a comment in the function body.
