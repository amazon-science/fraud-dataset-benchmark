


### Package Usage

```
from fdb.datasets import FraudDatasetBenchmark

# all_keys = ['fakejob', 'vechicleloan', 'malurl', 'ieeecis', 'ccfraud','fraudecom', 'twitterbot', 'ipblock'] 
key = 'ipblock'

obj = FraudDatasetBenchmark(key=key)
print(obj.key)

print('Train set: ')
display(obj.train.head())
print(len(obj.train.columns))
print(obj.train.shape)

print('Test set: ')
display(obj.test.head())
print(obj.test.shape)

print('Test scores')
display(obj.test_labels.head())
print(obj.test_labels['EVENT_LABEL'].value_counts())
print(obj.train['EVENT_LABEL'].value_counts(normalize=True))
print('=========')

``` 
