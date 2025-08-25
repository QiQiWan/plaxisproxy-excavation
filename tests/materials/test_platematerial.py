import unittest
from plaxisproxy_excavation.materials.platematerial import ElasticPlate, ElastoplasticPlate, PlateType

class TestElasticPlate(unittest.TestCase):
    def test_default_optional_parameters(self):
        """ElasticPlate with only required params; optional parameters default to 0."""
        plate = ElasticPlate(
            name="Plate1",
            type=PlateType.Elastic,
            comment="plate comment",
            gamma=7.0, E=150.0, nu=0.20,
            preventpunch=True, isotropic=True 
        )
        self.assertTrue(plate.preventpunch)
        self.assertTrue(plate.isotropic)
        self.assertEqual(plate.E2, 0)
        self.assertEqual(plate.G12, 12500000.0)
        self.assertEqual(plate.G13, 50E6)
        self.assertEqual(plate.G23, 50E6)
        self.assertEqual(repr(plate), "<plx.materials.elastic_plate>")

    def test_full_parameters(self):
        """ElasticPlate with all parameters explicitly provided."""
        plate = ElasticPlate(
            name="Plate2",
            type=PlateType.Elastic,
            comment="plate comment",
            gamma=7.0, E=150.0, nu=0.20,
            preventpunch=False, isotropic=False,
            E2=120.0, G12=60.0, G13=40.0, G23=30.0
        )
        self.assertFalse(plate.preventpunch)
        self.assertFalse(plate.isotropic)
        self.assertEqual(plate.E2, 120.0)
        self.assertEqual(plate.G12, 60.0)
        self.assertEqual(plate.G13, 40.0)
        self.assertEqual(plate.G23, 30.0)
        self.assertEqual(repr(plate), "<plx.materials.elastic_plate>")

class TestElastoplasticPlate(unittest.TestCase):
    def test_default_yield_parameters(self):
        """ElastoplasticPlate without yield parameters provided (should use defaults)."""
        plate = ElastoplasticPlate(
            name="Plate3",
            type=PlateType.Elastoplastic,
            comment="plate comment",
            gamma=9.0, E=180.0, nu=0.25,
            preventpunch=True, isotropic=True
        )
        self.assertEqual(plate.sigma_y_11, 40e3)
        self.assertEqual(plate.W_11, 0.05)
        self.assertEqual(plate.sigma_y_22, 40e3)
        self.assertEqual(plate.W_22, 0.04)

        self.assertTrue(plate.preventpunch)
        self.assertTrue(plate.isotropic)
        self.assertEqual(repr(plate), "<plx.materials.elastoplastic_plate>")

    def test_full_yield_parameters(self):
        """ElastoplasticPlate with all yield parameters specified."""
        plate = ElastoplasticPlate(
            name="Plate4",
            type=PlateType.Elastoplastic,
            comment="plate comment",
            gamma=9.0, E=180.0, nu=0.25,
            preventpunch=False, isotropic=False,
            E2=100.0, G12=50.0, G13=50.0, G23=50.0,
            sigma_y_11=300.0, W_11=2.0,
            sigma_y_22=200.0, W_22=1.5
        )
        self.assertEqual(plate.sigma_y_11, 300.0)
        self.assertEqual(plate.W_11, 2.0)
        self.assertEqual(plate.sigma_y_22, 200.0)
        self.assertEqual(plate.W_22, 1.5)
        self.assertFalse(plate.preventpunch)
        self.assertFalse(plate.isotropic)
        self.assertEqual(plate.E2, 100.0)
        self.assertEqual(plate.G12, 50.0)
        self.assertEqual(plate.G13, 50.0)
        self.assertEqual(plate.G23, 50.0)
        self.assertEqual(repr(plate), "<plx.materials.elastoplastic_plate>")

if __name__ == '__main__':
    unittest.main()
