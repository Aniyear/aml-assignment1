"""
Feature engineering module for water surface classification.
Calculates spectral indices and extracts features from satellite imagery.
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Handles feature engineering for satellite imagery."""
    
    @staticmethod
    def calculate_ndwi(green_band, nir_band):
        """
        Calculate Normalized Difference Water Index (NDWI).
        
        NDWI = (Green - NIR) / (Green + NIR)
        
        Parameters:
        -----------
        green_band : array-like
            Green band reflectance values
        nir_band : array-like
            Near-infrared band reflectance values
            
        Returns:
        --------
        np.ndarray
            NDWI values
        """
        denominator = green_band + nir_band
        denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)
        return (green_band - nir_band) / denominator
    
    @staticmethod
    def calculate_mndwi(green_band, swir_band):
        """
        Calculate Modified Normalized Difference Water Index (MNDWI).
        
        MNDWI = (Green - SWIR) / (Green + SWIR)
        
        Parameters:
        -----------
        green_band : array-like
            Green band reflectance values
        swir_band : array-like
            Short-wave infrared band reflectance values
            
        Returns:
        --------
        np.ndarray
            MNDWI values
        """
        denominator = green_band + swir_band
        denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)
        return (green_band - swir_band) / denominator
    
    @staticmethod
    def calculate_ndvi(red_band, nir_band):
        """
        Calculate Normalized Difference Vegetation Index (NDVI).
        
        NDVI = (NIR - Red) / (NIR + Red)
        
        Parameters:
        -----------
        red_band : array-like
            Red band reflectance values
        nir_band : array-like
            Near-infrared band reflectance values
            
        Returns:
        --------
        np.ndarray
            NDVI values
        """
        denominator = nir_band + red_band
        denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)
        return (nir_band - red_band) / denominator
    
    @staticmethod
    def calculate_evi(blue_band, red_band, nir_band):
        """
        Calculate Enhanced Vegetation Index (EVI).
        
        EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        
        Parameters:
        -----------
        blue_band : array-like
            Blue band reflectance values
        red_band : array-like
            Red band reflectance values
        nir_band : array-like
            Near-infrared band reflectance values
            
        Returns:
        --------
        np.ndarray
            EVI values
        """
        denominator = nir_band + 6 * red_band - 7.5 * blue_band + 1
        denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)
        return 2.5 * (nir_band - red_band) / denominator
    
    @staticmethod
    def create_features(data):
        """
        Create comprehensive feature set from spectral bands.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with spectral bands
            
        Returns:
        --------
        pd.DataFrame
            Enhanced dataset with calculated indices
        """
        df = data.copy()
        
        # Ensure we have the required bands
        required_bands = ['B2_Blue', 'B3_Green', 'B4_Red', 'B8_NIR', 'B11_SWIR1', 'B12_SWIR2']
        available_bands = [col for col in required_bands if col in df.columns]
        
        if len(available_bands) < 4:
            raise ValueError("Insufficient spectral bands for feature engineering")
        
        # Calculate indices if bands are available
        if 'B3_Green' in df.columns and 'B8_NIR' in df.columns:
            df['NDWI'] = FeatureEngineer.calculate_ndwi(
                df['B3_Green'].values, 
                df['B8_NIR'].values
            )
        
        if 'B3_Green' in df.columns and 'B11_SWIR1' in df.columns:
            df['MNDWI'] = FeatureEngineer.calculate_mndwi(
                df['B3_Green'].values, 
                df['B11_SWIR1'].values
            )
        
        if 'B4_Red' in df.columns and 'B8_NIR' in df.columns:
            df['NDVI'] = FeatureEngineer.calculate_ndvi(
                df['B4_Red'].values, 
                df['B8_NIR'].values
            )
        
        if all(band in df.columns for band in ['B2_Blue', 'B4_Red', 'B8_NIR']):
            df['EVI'] = FeatureEngineer.calculate_evi(
                df['B2_Blue'].values,
                df['B4_Red'].values,
                df['B8_NIR'].values
            )
        
        # Calculate band ratios
        if 'B8_NIR' in df.columns and 'B4_Red' in df.columns:
            df['NIR_Red_Ratio'] = df['B8_NIR'] / (df['B4_Red'] + np.finfo(float).eps)
        
        if 'B11_SWIR1' in df.columns and 'B8_NIR' in df.columns:
            df['SWIR_NIR_Ratio'] = df['B11_SWIR1'] / (df['B8_NIR'] + np.finfo(float).eps)
        
        return df

