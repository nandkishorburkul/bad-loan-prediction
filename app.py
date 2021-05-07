from flask import Flask, request, jsonify,render_template
import test_model
# import config
app = Flask(__name__)

@app.route('/')
def index():
   return render_template('income.html')

@app.route('/result',methods = ['POST'])
def result():
   if request.method == 'POST':
      data = request.form 
      result = test_model.rf_Model().get_loan_pred(data)
      if float(result) == 0.0:
         predictio = "Good Loan"
      else:
         prediction = 'Bad Loan'
      return render_template("result.html", prediction = prediction)
   


if __name__ == "__main__":
    app.run()