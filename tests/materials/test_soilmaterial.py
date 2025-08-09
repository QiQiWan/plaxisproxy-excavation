import unittest
from plaxisproxy_excavation.materials.soilmaterial import (
    BaseSoilMaterial, MCMaterial, MCCMaterial, HSSMaterial,
    SoilMaterialFactory, SoilMaterialsType, MCGWType, MCGwSWCC
)

class TestBaseSoilMaterial(unittest.TestCase):
    def test_initial_state_and_repr(self):
        """BaseSoilMaterial initialization sets properties and default flags correctly."""
        soil = BaseSoilMaterial("Soil1", SoilMaterialsType.MC, "soil comment",
                                gamma=10.0, E=100.0, nu=0.30, gamma_sat=11.0, e_init=0.5)

        self.assertEqual(soil.gamma, 10.0)
        self.assertEqual(soil.E, 100.0)
        self.assertEqual(soil.nu, 0.30)
        self.assertEqual(soil.gamma_sat, 11.0)
        self.assertEqual(soil.e_init, 0.5)

        self.assertFalse(soil._set_pore_stress)
        self.assertFalse(soil.set_ug_water)      
        self.assertFalse(soil._set_additional_interface)
        self.assertFalse(soil._set_additional_initial)
        self.assertFalse(hasattr(soil, "_pore_stress_type") or hasattr(soil, "_water_value"))
        # n_init = e_init / (1 + e_init)
        self.assertAlmostEqual(soil.n_init, 0.5 / 1.5, places=6)
        self.assertEqual(repr(soil), "<plx.materials.soilbase>")

    def test_set_super_pore_stress_parameters_default(self):
        """Default arguments for set_super_pore_stress_parameters set flags and values."""
        soil = BaseSoilMaterial("Soil2", SoilMaterialsType.MC, "soil comment",
                                gamma=9.0, E=90.0, nu=0.25, gamma_sat=10.0, e_init=0.4)
        soil.set_super_pore_stress_parameters()  

        self.assertTrue(getattr(soil, "_set_pore_stress", False))
        self.assertEqual(soil.pore_stress_type, 0)
        self.assertEqual(soil.vu, 0)
        self.assertIsNone(soil.water_value)

    def test_set_super_pore_stress_parameters_custom(self):
        """Custom values for set_super_pore_stress_parameters are stored correctly."""
        soil = BaseSoilMaterial("Soil3", SoilMaterialsType.MC, "soil comment",
                                gamma=9.5, E=95.0, nu=0.30, gamma_sat=10.5, e_init=0.6)
        soil.set_super_pore_stress_parameters(pore_stress_type=1, vu=1, value=123.45)

        self.assertTrue(soil._set_pore_stress)
        self.assertEqual(soil.pore_stress_type, 1)
        self.assertEqual(soil.vu, 1)
        self.assertEqual(soil.water_value, 123.45)

    def test_set_under_ground_water(self):
        """set_under_ground_water should set all underground water parameters and flag."""
        soil = BaseSoilMaterial("Soil4", SoilMaterialsType.MC, "soil comment",
                                gamma=8.0, E=80.0, nu=0.30, gamma_sat=9.0, e_init=0.4)
        soil.set_under_ground_water(
            type=MCGWType.Standard,
            SWCC_method=MCGwSWCC.Van,
            soil_posi="pos", soil_fine="fine",
            Gw_defaults={"key": "val"},
            infiltration=0.1,
            default_method="default",
            kx=1e-5, ky=2e-5, kz=3e-5,
            Gw_Psiunsat=0.05
        )

        self.assertTrue(soil.set_ug_water)
        self.assertEqual(soil.Gw_type, MCGWType.Standard)
        self.assertEqual(soil.SWCC_method, MCGwSWCC.Van)
        self.assertEqual(soil.soil_posi, "pos")
        self.assertEqual(soil.soil_fine, "fine")
        self.assertEqual(soil.Gw_defaults, {"key": "val"})
        self.assertEqual(soil.infiltration, 0.1)
        self.assertEqual(soil.default_method, "default")
        self.assertAlmostEqual(soil.kx, 1e-5, places=9)
        self.assertAlmostEqual(soil.ky, 2e-5, places=9)
        self.assertAlmostEqual(soil.kz, 3e-5, places=9)
        self.assertEqual(soil.Gw_Psiunsat, 0.05)

    def test_set_additional_interface_parameters(self):
        """set_additional_interface_parameters should set interface parameters and flag."""
        soil = BaseSoilMaterial("Soil5", SoilMaterialsType.MC, "soil comment",
                                gamma=11.0, E=110.0, nu=0.35, gamma_sat=12.0, e_init=0.7)
        soil.set_additional_interface_parameters(
            stiffness_define="stiff",
            strengthen_define="strong",
            k_n=1000.0, k_s=500.0,
            R_inter=0.8,
            gap_closure="gap",
            cross_permeability=0.001,
            drainage_conduct1=0.002,
            drainage_conduct2=0.003
        )
        self.assertTrue(soil._set_additional_interface)
        self.assertEqual(soil.stiffness_define, "stiff")
        self.assertEqual(soil.strengthen_define, "strong")
        self.assertEqual(soil.k_n, 1000.0)
        self.assertEqual(soil.k_s, 500.0)
        self.assertEqual(soil.R_inter, 0.8)
        self.assertEqual(soil.gap_closure, "gap")
        self.assertEqual(soil.cross_permeability, 0.001)
        self.assertEqual(soil.drainage_conduct1, 0.002)
        self.assertEqual(soil.drainage_conduct2, 0.003)

    def test_set_additional_initial_parameters(self):
        """set_additional_initial_parameters should set K0 parameters and flag."""
        soil = BaseSoilMaterial("Soil6", SoilMaterialsType.MC, "soil comment",
                                gamma=12.0, E=120.0, nu=0.40, gamma_sat=13.0, e_init=0.8)
        soil.set_additional_initial_parameters(K_0_define="K0Defined", K_0_x=0.6, K_0_y=0.7)
        self.assertTrue(soil._set_additional_initial)
        self.assertEqual(soil.K_0_define, "K0Defined")
        self.assertEqual(soil.K_0_x, 0.6)
        self.assertEqual(soil.K_0_y, 0.7)

class TestSoilMaterialSubclasses(unittest.TestCase):
    def test_mc_material_properties(self):
        """MCMaterial inherits BaseSoilMaterial and adds Mohr-Coulomb specific properties."""
        mc = MCMaterial(
            name="SoilMC",
            type=SoilMaterialsType.MC,
            comment="mc comment",
            gamma=10.0, E=100.0, nu=0.30,
            gamma_sat=11.0, e_init=0.5,
            E_ref=200.0, c_ref=5.0, phi=30.0, psi=5.0, tensile_strength=10.0
        )
        self.assertIsInstance(mc, BaseSoilMaterial)
        self.assertEqual(repr(mc), "<plx.materials.MCSoil>")

        self.assertEqual(mc.E_ref, 200.0)
        self.assertEqual(mc.c_ref, 5.0)
        self.assertEqual(mc.phi, 30.0)
        self.assertEqual(mc.psi, 5.0)
        self.assertEqual(mc.tensile_strength, 10.0)

        self.assertAlmostEqual(mc.n_init, 0.5 / 1.5, places=6)

    def test_mcc_material_properties(self):
        """MCCMaterial adds Modified Cam-Clay specific parameters."""
        mcc = MCCMaterial(
            name="SoilMCC",
            type=SoilMaterialsType.MCC,
            comment="mcc comment",
            gamma=11.0, E=110.0, nu=0.35,
            gamma_sat=12.0, e_init=0.6,
            lam=0.10, kar=0.05, M_CSL=1.2
        )
        self.assertIsInstance(mcc, BaseSoilMaterial)
        self.assertEqual(repr(mcc), "<plx.materials.MCCSoil>")
        self.assertEqual(mcc.lam, 0.10)
        self.assertEqual(mcc.kar, 0.05)
        self.assertEqual(mcc.M_CSL, 1.2)

    def test_hss_material_properties(self):
        """HSSMaterial adds Hardening Soil model parameters."""
        hss = HSSMaterial(
            name="SoilHSS",
            type=SoilMaterialsType.HSS,
            comment="hss comment",
            gamma=12.0, E=120.0, nu=0.30,
            gamma_sat=13.0, e_init=0.7,
            E_oed=300.0, E_ur=400.0, m=0.8, P_ref=100.0,
            G0=200.0, gamma_07=0.0001, c=10.0, phi=25.0, psi=0.0
        )
        self.assertIsInstance(hss, BaseSoilMaterial)
        self.assertEqual(repr(hss), "<plx.materials.HSSSoil>")

        self.assertEqual(hss.E_oed, 300.0)
        self.assertEqual(hss.E_ur, 400.0)
        self.assertEqual(hss.m, 0.8)
        self.assertEqual(hss.P_ref, 100.0)
        self.assertEqual(hss.G0, 200.0)
        self.assertEqual(hss.gamma_07, 0.0001)
        self.assertEqual(hss.c, 10.0)
        self.assertEqual(hss.phi, 25.0)
        self.assertEqual(hss.psi, 0.0)

class TestSoilMaterialFactory(unittest.TestCase):
    def test_create_mc_material_success(self):
        """SoilMaterialFactory.create creates MCMaterial when provided MC type and all params."""
        params = {
            "name": "SoilMC2",
            "comment": "mc",
            "gamma": 9.0, "E": 90.0, "nu": 0.30,
            "gamma_sat": 10.0, "e_init": 0.4,
            "E_ref": 180.0, "c_ref": 4.0, "phi": 28.0, "psi": 3.0, "tensile_strength": 8.0
        }
        mc = SoilMaterialFactory.create(SoilMaterialsType.MC, **params)
        self.assertIsInstance(mc, MCMaterial)
        # MCMaterial 应正确实例化并包含传入参数
        self.assertEqual(mc.name, "SoilMC2")
        self.assertEqual(mc.E_ref, 180.0)
        self.assertEqual(mc.phi, 28.0)

    def test_create_mc_material_missing_arg(self):
        """Missing required argument for MCMaterial should raise TypeError."""
        params = {
            "name": "MissingMC",
            # 缺少 tensile_strength 参数
            "gamma": 9.0, "E": 90.0, "nu": 0.30,
            "gamma_sat": 10.0, "e_init": 0.4,
            "E_ref": 180.0, "c_ref": 4.0, "phi": 28.0, "psi": 3.0
        }
        with self.assertRaises(TypeError) as context:
            SoilMaterialFactory.create(SoilMaterialsType.MC, **params)
        self.assertIn("Missing required argument for MCMaterial", str(context.exception))

    def test_create_mcc_material_success(self):
        """Factory creates MCCMaterial with correct parameters."""
        params = {
            "name": "SoilMCC2",
            "comment": "mcc",
            "gamma": 10.0, "E": 100.0, "nu": 0.30,
            "gamma_sat": 11.0, "e_init": 0.5,
            "lam": 0.11, "kar": 0.06, "M_CSL": 1.3
        }
        mcc = SoilMaterialFactory.create(SoilMaterialsType.MCC, **params)
        self.assertIsInstance(mcc, MCCMaterial)
        self.assertEqual(mcc.name, "SoilMCC2")
        self.assertEqual(mcc.M_CSL, 1.3)

    def test_create_mcc_material_missing_arg(self):
        """Missing required argument for MCCMaterial should raise TypeError."""
        params = {
            "name": "MissingMCC",
            "gamma": 10.0, "E": 100.0, "nu": 0.30,
            "gamma_sat": 11.0, "e_init": 0.5,
            "lam": 0.11, "kar": 0.06
        }
        with self.assertRaises(TypeError) as context:
            SoilMaterialFactory.create(SoilMaterialsType.MCC, **params)
        self.assertIn("Missing required argument for MCCMaterial", str(context.exception))

    def test_create_hss_material_success(self):
        """Factory creates HSSMaterial with correct parameters."""
        params = {
            "name": "SoilHSS2",
            "comment": "hss",
            "gamma": 11.0, "E": 110.0, "nu": 0.33,
            "gamma_sat": 12.0, "e_init": 0.6,
            "E_oed": 320.0, "E_ur": 420.0, "m": 0.85, "P_ref": 110.0,
            "G0": 220.0, "gamma_07": 0.0002, "c": 12.0, "phi": 26.0, "psi": 1.0
        }
        hss = SoilMaterialFactory.create(SoilMaterialsType.HSS, **params)
        self.assertIsInstance(hss, HSSMaterial)
        self.assertEqual(hss.name, "SoilHSS2")
        self.assertEqual(hss.c, 12.0)
        self.assertEqual(hss.phi, 26.0)
        self.assertAlmostEqual(hss.gamma_07, 0.0002, places=6)

    def test_create_hss_material_missing_arg(self):
        """Missing required argument for HSSMaterial should raise TypeError."""
        params = {
            "name": "MissingHSS",
            # 缺少 psi 参数
            "gamma": 11.0, "E": 110.0, "nu": 0.33,
            "gamma_sat": 12.0, "e_init": 0.6,
            "E_oed": 320.0, "E_ur": 420.0, "m": 0.85, "P_ref": 110.0,
            "G0": 220.0, "gamma_07": 0.0002, "c": 12.0, "phi": 26.0
        }
        with self.assertRaises(TypeError) as context:
            SoilMaterialFactory.create(SoilMaterialsType.HSS, **params)
        self.assertIn("Missing required argument for HSSMaterial", str(context.exception))

    def test_create_unknown_type(self):
        """Unknown material type should raise ValueError."""
        with self.assertRaises(ValueError) as context:
            SoilMaterialFactory.create("UNKNOWN_TYPE", name="Unknown",
                                       gamma=5.0, E=50.0, nu=0.20,
                                       gamma_sat=6.0, e_init=0.3)
        self.assertIn("Unknown soil material type", str(context.exception))

if __name__ == '__main__':
    unittest.main()