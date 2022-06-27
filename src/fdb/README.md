


### Package Usage

```
# keys = ['fakejob', 'vechicleloan', 'malurl', 'ieee', 'ccfraud','fraudecom', 'twitterbot']
from fdb.datasets import FraudBenchmarkDataset
dataset = FraudBenchmarkDataset(key = 'fakejob')

train_dataset = dataset.train  # pandas dataframe of train dataset
test_dataset = dataset.test  # pandas dataframe of test dataset

eval_metrics = dataset.eval(y_true, y_pred) # evaluator

print(dataset.automl_performance) # evaluation metrics on test data trained using  

``` 
