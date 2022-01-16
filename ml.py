import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
df = pd.read_csv('machine_learning.csv')
df.drop('ID', axis = 1, inplace = True)

def sex(s):
	if s == 'M':
		return 1
	if s == 'F':
		return 0
df['sex'] = df['sex'].apply(sex)

def adr(a):
	if a == 'Urban':
		return 1
	if a == 'Rural':
		return 0
df['address'] = df['address'].apply(adr)

def fam(f):
	if f == '3 persons or less':
		return 0
	if f == 'greater than 3 persons':
		return 1
df['famsize'] = df['famsize'].apply(fam)

def meduc(m):
	if m == 'none':
		return 0
	if m == 'primary education (4th grade)':
		return 1
	if m == '5th to 9th grade':
		return 2
	if m == 'higher education':
		return 3
	if m == 'secondary education':
		return 4
df['Medu'] = df['Medu'].apply(meduc)

def feduc(f):
	if f == 'none':
		return 0
	if f == 'primary education (4th grade)':
		return 1
	if f == '5th to 9th grade':
		return 2
	if f == 'higher education':
		return 3
	if f == 'secondary education':
		return 4
df['Fedu'] = df['Fedu'].apply(feduc)

def fmark(f):
	return f + '*'
df['Fjob'] = df['Fjob'].apply(fmark)

df[list(pd.get_dummies(df['Mjob']).columns)] = pd.get_dummies(df['Mjob'])
df.drop('Mjob', axis = 1, inplace = True)

df[list(pd.get_dummies(df['Fjob']).columns)] = pd.get_dummies(df['Fjob'])
df.drop('Fjob', axis = 1, inplace = True)

df[list(pd.get_dummies(df['reason']).columns)] = pd.get_dummies(df['reason'])
df.drop('reason', axis = 1, inplace = True)

df[list(pd.get_dummies(df['guardian']).columns)] = pd.get_dummies(df['guardian'])
df.drop('guardian', axis = 1, inplace = True)

def tt(t):
	if t == 'less than 15 min.':
		return 0
	if t == '15 to 30 min.':
		return 1
	if t == '30 min. to 1 hour':
		return 2
	if t == 'more than 1 hour':
		return 3
df['traveltime'] = df['traveltime'].apply(tt)

def st(t):
	if t == '2 to 5 hours':
		return 1
	if t == 'less than 2 hours':
		return 0
	if t == '5 to 10 hours':
		return 2
	if t == 'more than 10 hours':
		return 3
df['studytime'] = df['studytime'].apply(st)

def schsup(s):
	if s == 'no':
		return 0
	if s == 'yes':
		return 1
df['schoolsup'] = df['schoolsup'].apply(schsup)

def fsup(f):
	if f == 'no':
		return 0
	if f == 'yes':
		return 1
df['famsup'] = df['famsup'].apply(fsup)

def pay(p):
	if p == 'no':
		return 0
	if p == 'yes':
		return 1
df['paid'] = df['paid'].apply(pay)

def act(a):
	if a == 'no':
		return 0
	if a == 'yes':
		return 1
df['activities'] = df['activities'].apply(act)

def ft(f):
	if f == 'very low ':
		return 0
	if f == 'low':
		return 1
	if f == 'medium':
		return 2
	if f == 'high':
		return 3
	if f == 'very high':
		return 4
df['freetime'] = df['freetime'].apply(ft)

def fam(f):
	if f == 'very bad':
		return 0
	if f == 'bad':
		return 1
	if f == 'normal':
		return 2
	if f == 'good':
		return 3
	if f == 'excellent':
		return 4
df['famrel'] = df['famrel'].apply(fam)

def nur(n):
	if n == 'no':
		return 0
	if n == 'yes':
		return 1
df['nursery'] = df['nursery'].apply(nur)

def hig(h):
	if h == 'no':
		return 0
	if h == 'yes':
		return 1
df['higher'] = df['higher'].apply(hig)

def inter(i):
	if i == 'no':
		return 0
	if i == 'yes':
		return 1
df['internet'] = df['internet'].apply(inter)


df.dropna(inplace = True)
#df.info()
X = df.drop('result', axis = 1)
y = df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred) * 100)


print('__________________________________________')


df6 = pd.read_csv('test.csv')
df6.drop('ID', axis = 1, inplace = True)

def sex(s):
	if s == 'M':
		return 1
	if s == 'F':
		return 0
df6['sex'] = df6['sex'].apply(sex)

def adr(a):
	if a == 'Urban':
		return 1
	if a == 'Rural':
		return 0
df6['address'] = df6['address'].apply(adr)

def fam(f):
	if f == '3 persons or less':
		return 0
	if f == 'greater than 3 persons':
		return 1
df6['famsize'] = df6['famsize'].apply(fam)

def meduc(m):
	if m == 'none':
		return 0
	if m == 'primary education (4th grade)':
		return 1
	if m == '5th to 9th grade':
		return 2
	if m == 'higher education':
		return 3
	if m == 'secondary education':
		return 4
df6['Medu'] = df6['Medu'].apply(meduc)

def feduc(f):
	if f == 'none':
		return 0
	if f == 'primary education (4th grade)':
		return 1
	if f == '5th to 9th grade':
		return 2
	if f == 'higher education':
		return 3
	if f == 'secondary education':
		return 4
df6['Fedu'] = df6['Fedu'].apply(feduc)

def fmark(f):
	return f + '*'
df6['Fjob'] = df6['Fjob'].apply(fmark)

df6[list(pd.get_dummies(df6['Mjob']).columns)] = pd.get_dummies(df6['Mjob'])
df6.drop('Mjob', axis = 1, inplace = True)

df6[list(pd.get_dummies(df6['Fjob']).columns)] = pd.get_dummies(df6['Fjob'])
df6.drop('Fjob', axis = 1, inplace = True)

df6[list(pd.get_dummies(df6['reason']).columns)] = pd.get_dummies(df6['reason'])
df6.drop('reason', axis = 1, inplace = True)

df6[list(pd.get_dummies(df6['guardian']).columns)] = pd.get_dummies(df6['guardian'])
df6.drop('guardian', axis = 1, inplace = True)

def tt(t):
	if t == 'less than 15 min.':
		return 0
	if t == '15 to 30 min.':
		return 1
	if t == '30 min. to 1 hour':
		return 2
	if t == 'more than 1 hour':
		return 3
df6['traveltime'] = df6['traveltime'].apply(tt)

def st(t):
	if t == '2 to 5 hours':
		return 1
	if t == 'less than 2 hours':
		return 0
	if t == '5 to 10 hours':
		return 2
	if t == 'more than 10 hours':
		return 3
df6['studytime'] = df6['studytime'].apply(st)

def schsup(s):
	if s == 'no':
		return 0
	if s == 'yes':
		return 1
df6['schoolsup'] = df6['schoolsup'].apply(schsup)

def fsup(f):
	if f == 'no':
		return 0
	if f == 'yes':
		return 1
df6['famsup'] = df6['famsup'].apply(fsup)

def pay(p):
	if p == 'no':
		return 0
	if p == 'yes':
		return 1
df6['paid'] = df6['paid'].apply(pay)

def act(a):
	if a == 'no':
		return 0
	if a == 'yes':
		return 1
df6['activities'] = df6['activities'].apply(act)

def ft(f):
	if f == 'very low ':
		return 0
	if f == 'low':
		return 1
	if f == 'medium':
		return 2
	if f == 'high':
		return 3
	if f == 'very high':
		return 4
df6['freetime'] = df6['freetime'].apply(ft)

def fam(f):
	if f == 'very bad':
		return 0
	if f == 'bad':
		return 1
	if f == 'normal':
		return 2
	if f == 'good':
		return 3
	if f == 'excellent':
		return 4
df6['famrel'] = df6['famrel'].apply(fam)

def nur(n):
	if n == 'no':
		return 0
	if n == 'yes':
		return 1
df6['nursery'] = df6['nursery'].apply(nur)

def hig(h):
	if h == 'no':
		return 0
	if h == 'yes':
		return 1
df6['higher'] = df6['higher'].apply(hig)

def inter(i):
	if i == 'no':
		return 0
	if i == 'yes':
		return 1
df6['internet'] = df6['internet'].apply(inter)

df6['Medu'].fillna(3.0, inplace = True)
df6['Fedu'].fillna(1.0, inplace = True)
df6['freetime'].fillna(3.0, inplace = True)
X_test = sc.transform(df6)
y_pred = classifier.predict(X_test)
print(y_pred)
result=pd.DataFrame({'ID':ID, 'result':y_pred})
result.to_csv('my_result.csv',index=False)