import unittest
from scipy.optimize import linprog
from LP import score_2d

class TestLinearProgramming(unittest.TestCase):

    def test_objective_function_logic(self):
        """
        Tests the manual Objective Function Z = 4x + 3y from Part 1.
        """
        # Test Origin 0, 0 -> Should be 0
        self.assertEqual(score_2d(0, 0), 0)
        
        # Test Optimal Intersection 2, 4 -> 4*2 + 3*4 = 8 + 12 = 20
        self.assertEqual(score_2d(2, 4), 20)
        
        # Test Time Bound Corner 0, 6 -> 4*0 + 3*6 = 18
        self.assertEqual(score_2d(0, 6), 18)

    def test_constraint_validity(self):
        """
        Manually verifies if specific points fall within the Feasible Region constraints.
        Constraints:
          1. x + y <= 6
          2. 2x + y <= 8
        """
        # Point 2, 4 - The intersection
        x, y = 2, 4
        self.assertTrue(x + y <= 6, "Point 2, 4 should satisfy Time constraint")
        self.assertTrue(2*x + y <= 8, "Point 2, 4 should satisfy Energy constraint")

        # Point 5, 5 - Clearly out of bounds
        x_bad, y_bad = 5, 5
        self.assertFalse(x_bad + y_bad <= 6, "Point 5, 5 should fail Time constraint")

    def test_solver_integration(self):
        """
        Integration Test: Verifies scipy.optimize.linprog actually solves the 3-var problem.
        Replicates Part 2 logic to ensure library compatibility.
        """
        # Standard Form Minimization
        c = [-5.0, -4.0, -3.0]
        A_ub = [[1.0, 1.0, 1.0], [2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]
        b_ub = [30.0, 40.0, 30.0]
        bounds = [(0, None), (0, None), (0, None)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        # Assertions
        self.assertTrue(result.success, "Solver failed to find a solution")
        self.assertEqual(result.status, 0, "Solver status should be 0 Optimization terminated successfully")
        
        # Verify the result is not negative since we flip it back
        max_z = -result.fun
        self.assertGreater(max_z, 0, "Maximized objective value should be positive")

if __name__ == '__main__':
    unittest.main()