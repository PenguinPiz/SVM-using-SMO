#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>

using namespace std;

class simpified_SMO {

private:
  int m;
  double C;
  double tol;
  int max_passes;

  vector<vector<double>> support_vectors;
  vector<double> labels;
  vector<double> alphas;
  double b;

  //------------------------------Helper Methods-------------------------------
  double dotProduct(const vector<double> &a, const vector<double> &b) {

    if (a.size() != b.size()) {
      cerr << "Two Vectors are not same size" << endl;
      return 0;
    }

    double result = 0;
    for (size_t i = 0; i < a.size(); i++) {

      result += a[i] * b[i];
    }

    return result;
  }

  /**
   *
   * This method uses the dotProduct as a Kernel function as a simpler more
   * understandable function, this can be changed easily without any harm to
   * the algorithm
   *
   * @param x : function variable
   *
   */
  double evaluate_svm_function(const vector<double> &x) {

    double sum = 0.0;

    for (int i = 0; i < m; i++) {

      sum += alphas[i] * labels[i] * dotProduct(support_vectors[i], x);
    }

    return sum + b;
  }

public:
  /*
   * This is the contructor for the SVM that initializes every parameter listed
   * below
   *
   * @param C : regulization Parameter
   * @param tol : numerical tolerance
   * @param max_passes : max number of times to iterate over values of a's
   * without changing
   * @param support_vectors : training data vector (x0, x1, ... , xn)
   *               where x are same dimension vectors with training data
   * @param labels : tha individual classes (-1, +1) of the support_vectors
   *
   *
   *
   */

  simpified_SMO(const vector<vector<double>> &Support_vectors,
                const vector<double> &Labels, double C_param, double Tol,
                int Max_passes)
      : support_vectors(Support_vectors), labels(Labels), C(C_param), tol(Tol),
        max_passes(Max_passes) {
    m = support_vectors.size();
    alphas.assign(m, 0.0);
    b = 0.0;
  }

  /*
   * This method performs a simplified version of Sequential Minimal
   * Optimization using the dot Product as a Kernel function as mentioned above.
   * It sets the needed variables for the prediction later on.
   *
   * Information about this algorithm can be found in the mentioned sources.
   *
   */
  void train() {

    int passes = 0;

    random_device rd;
    mt19937 gen(rd());

    while (passes < max_passes) {

      int num_changed_alphas = 0;

      for (int i = 0; i < m; i++) {

        double E_i = evaluate_svm_function(support_vectors[i]) - labels[i];

        if ((labels[i] * E_i < -tol && alphas[i] < C) ||
            (labels[i] * E_i > tol && alphas[i] > 0)) {

          uniform_int_distribution<> dis(0, m - 2);
          int j = dis(gen);
          if (j >= i)
            j++;

          double E_j = evaluate_svm_function(support_vectors[j]) - labels[j];

          double alpha_i_old = alphas[i];
          double alpha_j_old = alphas[j];

          double L, H;

          if (labels[i] != labels[j]) {

            L = max(0.0, alphas[j] - alphas[i]);
            H = min(C, C + alphas[j] - alphas[i]);

          } else {

            L = max(0.0, alphas[i] + alphas[j] - C);
            H = min(C, alphas[i] + alphas[j]);
          }

          if (abs(L - H) < 1e-9)
            continue;

          double h = 2 * dotProduct(support_vectors[i], support_vectors[j]) -
                     dotProduct(support_vectors[i], support_vectors[i]) -
                     dotProduct(support_vectors[j], support_vectors[j]);

          if (h >= 0)
            continue;

          alphas[j] = alphas[j] - (labels[j] * (E_i - E_j)) / h;

          if (alphas[j] > H) {

            alphas[j] = H;

          } else if (alphas[j] < L) {

            alphas[j] = L;
          }

          if (abs(alphas[j] - alpha_j_old) < 10e-5)
            continue;

          alphas[i] =
              alphas[i] + labels[i] * labels[j] * (alpha_j_old - alphas[j]);

          double b1 = b - E_i -
                      labels[i] * (alphas[i] - alpha_i_old) *
                          dotProduct(support_vectors[i], support_vectors[i]) -
                      labels[j] * (alphas[j] - alpha_j_old) *
                          dotProduct(support_vectors[i], support_vectors[j]);

          double b2 = b - E_j -
                      labels[i] * (alphas[i] - alpha_i_old) *
                          dotProduct(support_vectors[i], support_vectors[j]) -
                      labels[j] * (alphas[j] - alpha_j_old) *
                          dotProduct(support_vectors[j], support_vectors[j]);

          if (alphas[i] > 0 && alphas[i] < C)
            b = b1;
          else if (alphas[j] > 0 && alphas[j] < C)
            b = b2;
          else
            b = (b1 + b2) / 2;

          num_changed_alphas++;
        }
      }

      if (num_changed_alphas == 0)
        passes++;
      else
        passes = 0;
    }
  }

  /*
   * This method performs the prediction for new Data vectors.
   *
   * @param x: Data Vector that has the same size as the support_vectors
   *
   * @return : Return either (-1) or (+1) that symbolizes the classification
   *
   */
  int predict(const vector<double> &x) {
    double result = evaluate_svm_function(x);
    return (result >= 0) ? 1 : -1;
  }

  /*
   *
   * Prints the two important parameters that are used for the hyperplane
   *
   */
  void printModelInfo() {
    cout << "Bias b: " << b << endl;
    cout << "Alphas: ";
    printVector(alphas);
  }

  void printVector(const vector<double> &vec) const {
    cout << '[';
    const char* sep = "";
    for (double v : vec) {
      cout << sep << v;
      sep = ", ";
    }
    cout << "]\n";
  }
  //--------------------Methods-------------------------------
};

/*
 *
 * Method to give option to read data from file (data.txt)
 *
 * file template :
 *
 * 5 //number of support vectors
 *
 * x1 y1 //values of vector now only 2
 * x2 y2
 * x3 y3
 * x4 y4
 * x5 y5
 *
 * label1 //class of above vector
 * label2
 * label3
 * label4
 * label5
 *
 * 2 //number of new data to be predicted
 *
 * x1 y1 //vector values of new data
 * x2 y2
 *
 */
bool readFromFile(vector<vector<double>> &X, vector<double> &y,
                  vector<vector<double>> &validationSet) {

  ifstream file("../data.txt");

  if (!file.is_open()) {

    cerr << "Could not open file" << endl;

    return false;
  }

  int n;
  file >> n;

  for (int i = 0; i < n; i++) {
    double x, y;
    file >> x;
    file >> y;
    X.push_back({x, y});
  }

  for (int i = 0; i < n; i++) {
    int label;
    file >> label;
    y.push_back(label);
  }

  int newDataSize;
  file >> newDataSize;

  for (int i = 0; i < newDataSize; i++) {
    double x, y;
    file >> x;
    file >> y;
    validationSet.push_back({x, y});
  }

  file.close();

  return true;
}

int main() {

  vector<vector<double>> X, testing_set;
  vector<double> y;

  if (!readFromFile(X, y, testing_set)) {

    cout << "Using template Data" << endl;
    // Training Vectors
    X = {
        {2, -1}, {3, -1},  {2, -2}, {3, -2}, {4, -2}, {3, -3}, {4, -3}, {0, -2},
        {1, -2}, {-1, -3}, {1, -3}, {2, -3}, {1, -4}, {2, -4}, {1, -5},
    };

    // Known classes for the above training vectors
    y = {1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1};

    // Validation Set to see if model prediction are accurate
    testing_set = {{-6, 6},   {1, -1}, {1.7, -2.6},
                   {2, -2.5}, {2, -4}, {2.8, -3.5}};
  }
  // Calling contructor to initialize parameters
  simpified_SMO svm(X, y, 1, 0.001, 20);

  // Training the model on Data given @X and @y
  cout << "Training SVM ... " << endl;
  svm.train();
  cout << "done!\n\n";

  svm.printModelInfo();

  // Performing prediction and printing results
  cout << "-------Predicting--------" << endl;
  for (size_t i = 0; i < testing_set.size(); i++) {
    int pred = svm.predict(testing_set[i]);
    cout << "Point (" << testing_set[i][0] << ", " << testing_set[i][1]
         << ") predicted class: (" << pred << ")" << endl
         << endl;
  }

  cout << "Sources : " << endl;
  cout << "https://cs229.stanford.edu/materials/smo.pdf" << endl;
  cout << "https://bitmask93.github.io/ml-blog/"
          "Sequential-Minimal-Optimization-for-Support-Vector-Machines/"
       << endl;
  cout << "https://en.wikipedia.org/wiki/Sequential_minimal_optimization"
       << endl;

  return 0;
}
