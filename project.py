from this import d
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import pickle

algorithms_accuracy = {"linear_regressor":[],"ridge_regressor":[],"svr_model":[],"random_forest":[]}

def module(X_train , X_test, y_train, y_test):
    standardization = StandardScaler()
    X_train_scaled = standardization.fit_transform(X_train)
    X_test_scaled = standardization.fit_transform(X_test)

    # Linear regression

    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train_scaled,y_train)
    y_pred_scaled = linear_regressor.predict(X_test_scaled)

    accuracy_linear_regression = r2_score(y_test,y_pred_scaled)
    algorithms_accuracy["linear_regressor"] = accuracy_linear_regression * 100

    # Ridge regression

    parameters = {'alpha':[1e-15,1e-3,1e-2,1,5,10,15]}
    ridge_regressor = Ridge()
    ridge = GridSearchCV(ridge_regressor,parameters,scoring='neg_mean_squared_error' ,cv=5)
    ridge.fit(X_train_scaled,y_train)
    y_pred_scale_ridge = ridge.predict(X_test_scaled)

    accuracy_ridge = r2_score(y_test,y_pred_scale_ridge)
    r2_score_result = {"r2_accuracy_ridge":accuracy_ridge}
    algorithms_accuracy["ridge_regressor"] = accuracy_ridge * 100

    # SVR

    svr_model = SVR()
    svr_model.fit(X_train_scaled,y_train)
    y_pred_svr = svr_model.predict(X_test_scaled)
    accuracy_svr = r2_score(y_test,y_pred_svr)
    algorithms_accuracy["svr_model"] = accuracy_svr * 100

    # Random Forest

    random_forest = RandomForestRegressor(n_estimators=2)
    random_forest.fit(X_train,y_train)
    accuracy_random_forest = random_forest.score(X_test,y_test)
    algorithms_accuracy["random_forest"] = accuracy_random_forest * 100

    return algorithms_accuracy

def find_best_module(get_module):
    best_accuracy_module = max(get_module , key=get_module.get)
    return best_accuracy_module

df = pd.read_csv('airfoil_self_noise.dat',sep = "\t" , header=None)
df.columns = ["Frequency" ,"Angle of attack", "Chord length", "Free-stream velocity","Suction side","sound pressure level"] 
X = df.iloc[:,:-1] # Independent Veriable
y = df.iloc[:,-1]  # Dependent Veriable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# for i in X_train.columns:
#   sns.boxplot(X_train[i])

unique_variables = len(df["sound pressure level"].unique()) # to check weather the record have unique or not

get_module = module(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)
best_accuracy_module = find_best_module(get_module)
print(best_accuracy_module)



# Create pickle file

pickle.dump(best_accuracy_module,open('model1.pkl','wb'))
pickled_model = pickle.load(open('model.pkl','rb'))