"""
Unit tests for hemisphere_strategy module.
"""
import pytest
import numpy as np
from falwa.hemisphere_strategy import (
    GlobalHemisphereStrategy,
    NorthernHemisphereOnlyStrategy,
    create_hemisphere_strategy,
    HemisphereStrategy
)


class TestGlobalStrategy:
    """Tests for GlobalHemisphereStrategy."""

    @pytest.fixture
    def strategy(self):
        return GlobalHemisphereStrategy()

    def test_should_compute_shem(self, strategy):
        assert strategy.should_compute_shem() is True

    def test_get_lat_dim(self, strategy):
        assert strategy.get_lat_dim(nlat_analysis=181, equator_idx=91) == 181
        assert strategy.get_lat_dim(nlat_analysis=121, equator_idx=61) == 121

    def test_is_northern_only(self, strategy):
        assert strategy.is_northern_only is False

    def test_get_ylat_for_ref_states(self, strategy):
        input_ylat = np.linspace(-90, 90, 181)
        internal_ylat = np.linspace(-90, 90, 181)
        result = strategy.get_ylat_for_ref_states(
            input_ylat, internal_ylat, nlat=181, nlat_analysis=181, need_interp=False
        )
        np.testing.assert_array_equal(result, input_ylat)

    def test_get_ylat_for_analysis(self, strategy):
        internal_ylat = np.linspace(-90, 90, 181)
        result = strategy.get_ylat_for_analysis(internal_ylat, nlat_analysis=181)
        np.testing.assert_array_equal(result, internal_ylat)

    def test_get_interp_slice(self, strategy):
        internal_ylat = np.linspace(-90, 90, 181)
        input_ylat = np.linspace(-90, 90, 180)
        source, target = strategy.get_interp_slice(
            internal_ylat, input_ylat, nlat=180, nlat_analysis=181
        )
        np.testing.assert_array_equal(source, internal_ylat)
        np.testing.assert_array_equal(target, input_ylat)

    def test_get_clat_slice(self, strategy):
        clat = np.cos(np.deg2rad(np.linspace(-90, 90, 181)))
        result = strategy.get_clat_slice(clat, equator_idx=91)
        np.testing.assert_array_equal(result, clat)

    def test_slice_3d_for_flux(self, strategy):
        slices = strategy.slice_3d_for_flux(equator_idx=91, nlat_analysis=181)
        assert slices == (slice(None), slice(None), slice(None))

    def test_get_ptref_for_stretch(self, strategy):
        ptref = np.ones((181, 49))
        result = strategy.get_ptref_for_stretch(
            ptref, nlat_analysis=181, kmax=49, equator_idx=91
        )
        np.testing.assert_array_equal(result, ptref)


class TestNorthernOnlyStrategy:
    """Tests for NorthernHemisphereOnlyStrategy."""

    @pytest.fixture
    def strategy(self):
        return NorthernHemisphereOnlyStrategy()

    def test_should_compute_shem(self, strategy):
        assert strategy.should_compute_shem() is False

    def test_get_lat_dim(self, strategy):
        assert strategy.get_lat_dim(nlat_analysis=181, equator_idx=91) == 91
        assert strategy.get_lat_dim(nlat_analysis=121, equator_idx=61) == 61

    def test_is_northern_only(self, strategy):
        assert strategy.is_northern_only is True

    def test_get_ylat_for_ref_states_with_interp(self, strategy):
        input_ylat = np.linspace(-90, 90, 180)  # Even grid, needs interpolation
        internal_ylat = np.linspace(-90, 90, 181)
        result = strategy.get_ylat_for_ref_states(
            input_ylat, internal_ylat, nlat=180, nlat_analysis=181, need_interp=True
        )
        expected = input_ylat[-(180 // 2):]  # Last 90 points
        np.testing.assert_array_equal(result, expected)

    def test_get_ylat_for_ref_states_without_interp(self, strategy):
        input_ylat = np.linspace(-90, 90, 181)  # Odd grid, no interpolation
        internal_ylat = np.linspace(-90, 90, 181)
        result = strategy.get_ylat_for_ref_states(
            input_ylat, internal_ylat, nlat=181, nlat_analysis=181, need_interp=False
        )
        expected = input_ylat[-(181 // 2 + 1):]  # Last 91 points
        np.testing.assert_array_equal(result, expected)

    def test_get_ylat_for_analysis(self, strategy):
        internal_ylat = np.linspace(-90, 90, 181)
        result = strategy.get_ylat_for_analysis(internal_ylat, nlat_analysis=181)
        expected = internal_ylat[-(181 // 2 + 1):]  # Last 91 points (0 to 90)
        np.testing.assert_array_equal(result, expected)

    def test_get_interp_slice(self, strategy):
        internal_ylat = np.linspace(-90, 90, 181)
        input_ylat = np.linspace(-90, 90, 180)
        source, target = strategy.get_interp_slice(
            internal_ylat, input_ylat, nlat=180, nlat_analysis=181
        )
        expected_source = internal_ylat[-(181 // 2 + 1):]
        expected_target = input_ylat[-(180 // 2):]
        np.testing.assert_array_equal(source, expected_source)
        np.testing.assert_array_equal(target, expected_target)

    def test_get_clat_slice(self, strategy):
        clat = np.cos(np.deg2rad(np.linspace(-90, 90, 181)))
        result = strategy.get_clat_slice(clat, equator_idx=91)
        expected = clat[-91:]  # NH portion only
        np.testing.assert_array_equal(result, expected)

    def test_slice_3d_for_flux(self, strategy):
        slices = strategy.slice_3d_for_flux(equator_idx=91, nlat_analysis=181)
        assert slices == (slice(None), slice(90, 181), slice(None))

    def test_get_ptref_for_stretch(self, strategy):
        ptref = np.ones((91, 49))  # NH-only ptref
        result = strategy.get_ptref_for_stretch(
            ptref, nlat_analysis=181, kmax=49, equator_idx=91
        )
        assert result.shape == (181, 49)
        # SH portion should be zeros
        np.testing.assert_array_equal(result[:90, :], np.zeros((90, 49)))
        # NH portion should be the original ptref
        np.testing.assert_array_equal(result[-91:, :], ptref)


class TestFactoryFunction:
    """Tests for create_hemisphere_strategy factory function."""

    def test_creates_global_strategy(self):
        strategy = create_hemisphere_strategy(False)
        assert isinstance(strategy, GlobalHemisphereStrategy)
        assert isinstance(strategy, HemisphereStrategy)

    def test_creates_northern_only_strategy(self):
        strategy = create_hemisphere_strategy(True)
        assert isinstance(strategy, NorthernHemisphereOnlyStrategy)
        assert isinstance(strategy, HemisphereStrategy)

    def test_strategies_have_opposite_behaviors(self):
        global_strategy = create_hemisphere_strategy(False)
        nh_only_strategy = create_hemisphere_strategy(True)

        assert global_strategy.should_compute_shem() is True
        assert nh_only_strategy.should_compute_shem() is False

        assert global_strategy.is_northern_only is False
        assert nh_only_strategy.is_northern_only is True

        # Same inputs should give different lat_dim
        assert global_strategy.get_lat_dim(181, 91) == 181
        assert nh_only_strategy.get_lat_dim(181, 91) == 91


class TestStrategyConsistency:
    """Tests to verify strategy behaviors are consistent with original code logic."""

    def test_lat_dim_matches_original_logic(self):
        """Verify get_lat_dim matches: equator_idx if nh_only else nlat_analysis"""
        nlat_analysis = 181
        equator_idx = 91

        # Original logic for northern_hemisphere_results_only=True
        expected_nh_only = equator_idx
        # Original logic for northern_hemisphere_results_only=False
        expected_global = nlat_analysis

        nh_strategy = NorthernHemisphereOnlyStrategy()
        global_strategy = GlobalHemisphereStrategy()

        assert nh_strategy.get_lat_dim(nlat_analysis, equator_idx) == expected_nh_only
        assert global_strategy.get_lat_dim(nlat_analysis, equator_idx) == expected_global

    def test_clat_slice_matches_original_logic(self):
        """Verify get_clat_slice matches original conditional slicing."""
        ylat = np.linspace(-90, 90, 181)
        clat = np.abs(np.cos(np.deg2rad(ylat)))
        equator_idx = 91

        # Original: clat[-equator_idx:] if nh_only else clat
        expected_nh_only = clat[-equator_idx:]
        expected_global = clat

        nh_strategy = NorthernHemisphereOnlyStrategy()
        global_strategy = GlobalHemisphereStrategy()

        np.testing.assert_array_equal(
            nh_strategy.get_clat_slice(clat, equator_idx), expected_nh_only
        )
        np.testing.assert_array_equal(
            global_strategy.get_clat_slice(clat, equator_idx), expected_global
        )

