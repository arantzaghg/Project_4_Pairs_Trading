import numpy as np

class KalmanFilterReg():
    def __init__(self, Q_filter: float = 0.01, R_filter: float = 10, P_filter: float = 0.1):
        """
        Initialize the Kalman Filter for linear regression.
        
        Parameters:
        Q_filter : float: Process noise covariance.
        R_filter : float: Measurement noise covariance.
        P_filter : float: Initial estimation error covariance.
        
        """
        
        self.w = np.zeros(2)

        # Transaction Matrix 
        self.A = np.eye(2)  

        # Noise in estimations
        self.Q = np.eye(2) * Q_filter

        # Noise in observations
        self.R = np.array([[1]]) * R_filter

        # Error in covariance prediction
        self.P = np.eye(2) * P_filter

    def predict(self):
        """
        Predict the next state.
        """
        
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self,x, y, vecm = None):
        """
        Update the Kalman Filter with new observations.

        Parameters:
        x : float: The independent variable observation.
        y : float: The dependent variable observation.
        vecm : float or None: Optional vector error correction model value.
        """

        if vecm is not None:
            y_n = vecm  
            C = np.array([[x, y]]) 
        
        else:
            y_n = y
            C = np.array([[1, x]])
        
        y_pred = C @ self.w

        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.inv(S)

        # Update error in covariance prediction
        self.P = (np.eye(2) - K @ C) @ self.P

        # Update estimations => x_t| t
        self.w = self.w + (K.flatten() * (y_n - y_pred))    
        
    @property
    def params(self):
        """
        Get the current parameters of the Kalman Filter.
        
        Returns:
        tuple: The current parameters (intercept and slope).
        """

        return self.w[0], self.w[1]



