#random forests
rfc = RandomForestClassifier(n_estimators=10)

####places########80.07%
X = train_matrix
y = train.places_target
results= rfc.fit(X, y).predict(test_matrix).astype(object)
target = np.asarray(test.places_target)
match = 0
for i in range(len(test)):
	print(i)
	match = match + difflib.SequenceMatcher(None,target[i],results[i]).ratio()
	print(match)

print(float(match)/len(test))

#######orgs########## 96.9%
X = train_matrix
y = train.orgs_target
results= rfc.fit(X, y).predict(test_matrix).astype(object)
target = np.asarray(test.orgs_target)
match = 0
for i in range(len(test)):
	print(i)
	match = match + difflib.SequenceMatcher(None,target[i],results[i]).ratio()
	print(match)

print(float(match)/len(test))

#####exchanes##########98.64%
X = train_matrix
y = train.exchanges_target
results= rfc.fit(X, y).predict(test_matrix).astype(object)
target = np.asarray(test.exchanges_target)
match = 0
for i in range(len(test)):
	print(i)
	match = match + difflib.SequenceMatcher(None,target[i],results[i]).ratio()
	print(match)

print(float(match)/len(test))

####people###########96.9%
X = train_matrix
y = train.people_target
results= rfc.fit(X, y).predict(test_matrix).astype(object)
target = np.asarray(test.people_target)
match = 0
for i in range(len(test)):
	print(i)
	match = match + difflib.SequenceMatcher(None,target[i],results[i]).ratio()
	print(match)

print(float(match)/len(test))

######topics######## 98.59%
X = train_matrix
y = train.exchanges_target
results= rfc.fit(X, y).predict(test_matrix).astype(object)
target = np.asarray(test.exchanges_target)
match = 0
for i in range(len(test)):
	print(i)
	match = match + difflib.SequenceMatcher(None,target[i],results[i]).ratio()
	print(match)

print(float(match)/len(test))
