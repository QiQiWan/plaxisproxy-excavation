import unittest
from plaxisproxy_excavation.materials.pilematerial import ElasticPile, ElastoplasticPile, LaterialResistanceType
from plaxisproxy_excavation.materials.beammaterial import BeamType

class TestElasticPile(unittest.TestCase):
    def test_properties_and_repr(self):
        """ElasticPile extends ElasticBeam with lateral resistance attributes."""
        pile = ElasticPile(
            name="Pile1",
            type=BeamType.Elastic,
            comment="pile comment",
            gamma=8.0, E=200.0, nu=0.30,
            cross_section="Cylinder", len1=1.5, len2=1.5,
            laterial_type=LaterialResistanceType.Linear,
            fric_table_Tmax=100.0, F_max=500.0
        )
        self.assertEqual(pile.cross_section, "Cylinder")
        self.assertEqual(pile.len1, 1.5)
        self.assertEqual(pile.len2, 1.5)
        self.assertEqual(pile.laterial_type, LaterialResistanceType.Linear)
        self.assertEqual(pile.fric_table_Tmax, 100.0)
        self.assertEqual(pile.F_max, 500.0)
        self.assertEqual(repr(pile), "<plx.materials.elastic_pile>")

class TestElastoplasticPile(unittest.TestCase):
    def test_properties_and_repr(self):
        """ElastoplasticPile combines ElasticPile and ElastoplasticBeam properties."""
        pile = ElastoplasticPile(
            name="Pile2",
            type=BeamType.Elastoplastic,
            comment="pile comment",
            gamma=10.0, E=250.0, nu=0.35,
            cross_section="Rectangle", len1=2.0, len2=2.5,
            laterial_type=LaterialResistanceType.MultiLinear,
            fric_table_Tmax=150.0, F_max=600.0,
            sigma_y=300.0, yield_dir="Y"
        )
        self.assertEqual(pile.laterial_type, LaterialResistanceType.MultiLinear)
        self.assertEqual(pile.fric_table_Tmax, 150.0)
        self.assertEqual(pile.F_max, 600.0)
        self.assertEqual(pile.sigma_y, 300.0)
        self.assertEqual(pile.yield_dir, "Y")
        self.assertEqual(pile.cross_section, "Rectangle")
        self.assertEqual(repr(pile), "<plx.materials.elastoplastic_pile>")

if __name__ == '__main__':
    unittest.main()
