import unittest
from plaxisproxy_excavation.materials.anchormaterial import ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor, AnchorType

class TestElasticAnchor(unittest.TestCase):
    def test_properties_and_repr(self):
        """ElasticAnchor stores type and EA correctly."""
        anchor = ElasticAnchor(name="Anchor1", type=AnchorType.Elastic, comment="anchor comment", EA=1000.0)
        self.assertEqual(anchor.type, AnchorType.Elastic)
        self.assertEqual(anchor.EA, 1000.0)
        self.assertEqual(repr(anchor), "<plx.materials.elastic_anchor>")

class TestElastoplasticAnchor(unittest.TestCase):
    def test_properties_and_repr(self):
        """ElastoplasticAnchor extends ElasticAnchor with F_max values."""
        anchor = ElastoplasticAnchor(
            name="Anchor2",
            type=AnchorType.Elastoplastic,
            comment="anchor comment",
            EA=1200.0,
            F_max_tens=300.0, F_max_comp=400.0
        )
        self.assertEqual(anchor.type, AnchorType.Elastoplastic)
        self.assertEqual(anchor.EA, 1200.0)

        self.assertEqual(anchor.F_max_tens, 300.0)
        self.assertEqual(anchor.F_max_comp, 400.0)
        self.assertEqual(repr(anchor), "<plx.materials.elastoplastic_anchor>")

class TestElastoPlasticResidualAnchor(unittest.TestCase):
    def test_properties_and_repr(self):
        """ElastoPlasticResidualAnchor extends ElastoplasticAnchor with residual forces."""
        anchor = ElastoPlasticResidualAnchor(
            name="Anchor3",
            type=AnchorType.ElastoPlasticResidual,
            comment="anchor comment",
            EA=1500.0,
            F_max_tens=350.0, F_max_comp=450.0,
            F_res_tens=50.0, F_res_comp=60.0
        )
        self.assertEqual(anchor.type, AnchorType.ElastoPlasticResidual)
        self.assertEqual(anchor.EA, 1500.0)
        self.assertEqual(anchor.F_max_tens, 350.0)
        self.assertEqual(anchor.F_max_comp, 450.0)
        self.assertEqual(anchor.F_res_tens, 50.0)
        self.assertEqual(anchor.F_res_comp, 60.0)
        self.assertEqual(repr(anchor), "<plx.materials.elastoplastic_residual_anchor>")

if __name__ == '__main__':
    unittest.main()
