import pickle
import json
import numpy as np
file_path = 'bad_loan_pred_forest.pkl'
model = pickle.load(open(file_path,'rb'))

__column_list = None

class rf_Model():
    def get_column_name(self):
        with open("D:/Python/Class Practice/Machine learning/My_Work_space_git/Bank_loan_project-1/Model/Name_of_column.json","r") as f:
            column_list = json.load(f)["columns_name"]
        return column_list

    def get_loan_pred(self,loan_amnt,term,rate,emp_lnt,home_os,annual_inc,purpose,addrs,DTI,delinq_2yrs,revol_util,total_acc,longest_lenths,verif_status):
        with open("D:/Python/Class Practice/Machine learning/My_Work_space_git/Bank_loan_project-1/Model/Label_list of addr_state.json", "r") as f:
                addrs_state = json.load(f)['addr_state']
        with open("D:/Python/Class Practice/Machine learning/My_Work_space_git/Bank_loan_project-1/Model/Label_list of home_ownership.json", "r") as Q:
                home_ownership = json.load(Q)['home_ownership']
        with open("D:/Python/Class Practice/Machine learning/My_Work_space_git/Bank_loan_project-1/Model/Label_list of purpose.json", "r") as W:
                purpose_1 = json.load(W)['purpose']
        with open("D:/Python/Class Practice/Machine learning/My_Work_space_git/Bank_loan_project-1/Model/Label_list of term.json", "r") as R:
                term_1 = json.load(R)['term']
        with open("D:/Python/Class Practice/Machine learning/My_Work_space_git/Bank_loan_project-1/Model/Label_list of verification_status.json", "r") as Z:
                verification_status = json.load(Z)['verification_status']
        with open("D:/Python/Class Practice/Machine learning/My_Work_space_git/Bank_loan_project-1/Model/Name_of_column.json","r") as f:
            column_list = json.load(f)["columns_name"]
        a = np.zeros(len(column_list))
        a[0] = loan_amnt
        a[1] = term_1.index(term)
        a[2] = float(rate)
        a[3] = float(emp_lnt)
        a[4] = home_ownership.index(home_os)
        a[5] = float(annual_inc)
        a[6] = purpose_1.index(purpose)
        a[7] = addrs_state.index(addrs)
        a[8] = float(DTI)
        a[9] = float(delinq_2yrs)
        a[10]= float(revol_util)
        a[11]= float(total_acc)
        a[12]= float(longest_lenths)
        a[13]= verification_status.index(verif_status)

        return model.predict([a])[0] 

if __name__ == "__main__":
    rf = rf_Model()
    price  = rf.get_loan_pred(5000,'36 months',10.65,10.0,'RENT',24000.0,'credit_card','AZ',27.65,0.0,83.7,9.0,26.0,'verified')
    print("The Predicted price is :",price)
