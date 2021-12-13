#include <eigen3/Eigen/Dense>

class KalmanFilter {

    // Declare all relevant matrices
    Eigen::MatrixXd F;  // State propagation matrix
    Eigen::MatrixXd B;  // System noise dynamic
    Eigen::MatrixXd Q;  // System/Process noise covariance matrix
    Eigen::MatrixXd H;  // Observation matrix
    Eigen::MatrixXd R;  // Measurement noise covariance matrix
    Eigen::MatrixXd P;  // Error covariance matrix
    Eigen::MatrixXd K;  // Kalman gain
    Eigen::MatrixXd P0; // Initial Error covariance matrix

    // System dimensions
    int m;  // Observation dimension
    int n;  // System state dimension

    // time members
    double t0;  // initial time
    double t;   // current time
    double dt;  // discrete time step

    // flags
    bool initialized;   // flag if filter is initialized
    
    // state estimates
    Eigen::VectorXd x_hat;
    Eigen::VectorXd x_hat_new;

    // set an identity matrix as required in update step
    Eigen::MatrixXd I;


    public:
        KalmanFilter();                 // empty constructor
        KalmanFilter(                   // full constructor
            double dt, 
            const Eigen::MatrixXd& F,
            const Eigen::MatrixXd& B,
            const Eigen::MatrixXd& Q,
            const Eigen::MatrixXd& H,
            const Eigen::MatrixXd& R,
            const Eigen::MatrixXd& P
            );
        void init();                                        // initialize with zeros
        void init(double t0, const Eigen::VectorXd& x0);    // init with initial guess on state

        void predict();                             // prediction step: soley based on system dynamic
        void predict(const Eigen::VectorXd& u);      // prediction step: with control input
        void update(const Eigen::VectorXd& y);      // update step: correct prediction based on measurements
        void filter_step(const Eigen::VectorXd& y); // combine prediction + update
        void filter_step(const Eigen::VectorXd& y, const Eigen::VectorXd& u);

        // Some getter functions
        Eigen::VectorXd get_state() { return x_hat; };
        double get_time() { return t; };


};