import unittest
from plaxisproxy_excavation.materials.beammaterial import ElasticBeam, ElastoplasticBeam, BeamType

class TestElasticBeam(unittest.TestCase):
    def test_properties_and_repr(self):
        """ElasticBeam stores cross-section and dimensions correctly."""
        beam = ElasticBeam(
            name="Beam1",
            type=BeamType.Elastic,
            comment="beam comment",
            gamma=5.0, E=100.0, nu=0.25,
            cross_section="Circular", len1=2.0, len2=3.0
        )
        # 检查自身属性
        self.assertEqual(beam.cross_section, "Circular")
        self.assertEqual(beam.len1, 2.0)
        self.assertEqual(beam.len2, 3.0)
        self.assertEqual(beam.gamma, 5.0)
        self.assertEqual(beam.type, BeamType.Elastic)
        self.assertEqual(repr(beam), "<plx.materials.elastic_beam>")

class TestElastoplasticBeam(unittest.TestCase):
    def test_properties_and_repr(self):
        """ElastoplasticBeam extends ElasticBeam with yield parameters."""
        beam = ElastoplasticBeam(
            name="Beam2",
            type=BeamType.Elastoplastic,
            comment="beam comment",
            gamma=6.0, E=120.0, nu=0.30,
            cross_section="Rectangle", len1=1.0, len2=4.0,
            sigma_y=250.0, yield_dir=1
        )
        self.assertEqual(beam.cross_section, "Rectangle")
        self.assertEqual(beam.len1, 1.0)
        self.assertEqual(beam.len2, 4.0)
        self.assertEqual(beam.sigma_y, 250.0)
        self.assertEqual(beam.yield_dir, 1)
        self.assertEqual(beam.gamma, 6.0)
        self.assertEqual(beam.type, BeamType.Elastoplastic)
        self.assertEqual(repr(beam), "<plx.materials.elastoplastic_beam>")

if __name__ == '__main__':
    unittest.main()