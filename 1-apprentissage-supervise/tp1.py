from functions import *




if __name__ == '__main__':
    
    dataset = pd.read_csv('credit_scoring.csv',sep=";")

    labels = "Seniority;Home;Time;Age;Marital;Records;Job;Expenses;Income;Assets;Debt;Amount;Price"
    labels = labels.split(";")

    # iloc[:,:-1] ":" : tout le champ
    x = dataset.iloc[:,:-1].values

    y = dataset.iloc[:,-1].values

    # Visualisation
    
    visualisation_hist(y, "x_status")
    
    x = normalisation(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7)        
    
    print_accuracy_tree_classifier(x_train, x_test, y_train, y_test)
    print_accuracy_knn(x_train, x_test, y_train, y_test, n=5)









