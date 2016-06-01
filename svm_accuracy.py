#one_vs_rest : accuracy

one_vs_rest = OneVsRestClassifier(LinearSVC(random_state=0))
####places########88.3%
X = train_matrix
y = train.places_target
results= one_vs_rest.fit(X, y).predict(test_matrix).astype(object)
target = np.asarray(test.places_target)
match = 0
for i in range(len(test)):
	print(i)
	match = match + difflib.SequenceMatcher(None,target[i],results[i]).ratio()
	print(match)

print(float(match)/len(test))

#######orgs########## 97%
X = train_matrix
y = train.orgs_target
results= one_vs_rest.fit(X, y).predict(test_matrix).astype(object)
target = np.asarray(test.orgs_target)
match = 0
for i in range(len(test)):
	print(i)
	match = match + difflib.SequenceMatcher(None,target[i],results[i]).ratio()
	print(match)

print(float(match)/len(test))

#####exchanes##########98.9%
X = train_matrix
y = train.exchanges_target
results= one_vs_rest.fit(X, y).predict(test_matrix).astype(object)
target = np.asarray(test.exchanges_target)
match = 0
for i in range(len(test)):
	print(i)
	match = match + difflib.SequenceMatcher(None,target[i],results[i]).ratio()
	print(match)

print(float(match)/len(test))

####people###########97.8%
X = train_matrix
y = train.people_target
results= one_vs_rest.fit(X, y).predict(test_matrix).astype(object)
target = np.asarray(test.people_target)
match = 0
for i in range(len(test)):
	print(i)
	match = match + difflib.SequenceMatcher(None,target[i],results[i]).ratio()
	print(match)

print(float(match)/len(test))

######topics######## 98.9%
X = train_matrix
y = train.exchanges_target
results= one_vs_rest.fit(X, y).predict(test_matrix).astype(object)
target = np.asarray(test.exchanges_target)
match = 0
for i in range(len(test)):
	print(i)
	match = match + difflib.SequenceMatcher(None,target[i],results[i]).ratio()
	print(match)

print(float(match)/len(test))
