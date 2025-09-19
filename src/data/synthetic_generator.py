# src/data/synthetic_generator.py

import pandas as pd
import numpy as np
from faker import Faker
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class SyntheticLendingDataGenerator:
    """
    Generate synthetic lending data with configurable bias patterns
    for testing fair lending compliance systems.
    """
    
    def __init__(self, n_samples: int = 10000, random_seed: int = 42):
        """
        Initialize the synthetic data generator.
        
        Args:
            n_samples: Number of samples to generate
            random_seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.faker = Faker()
        Faker.seed(random_seed)
        
        # Define demographic distributions (based on US census approximations)
        self.demographics = {
            'race': {
                'White': 0.60,
                'Black': 0.13,
                'Hispanic': 0.18,
                'Asian': 0.06,
                'Other': 0.03
            },
            'gender': {
                'Male': 0.49,
                'Female': 0.49,
                'Non-binary': 0.02
            },
            'age_groups': {
                '18-24': 0.10,
                '25-34': 0.20,
                '35-44': 0.25,
                '45-54': 0.20,
                '55-64': 0.15,
                '65+': 0.10
            }
        }
        
        # Geographic regions with associated economic factors
        self.regions = {
            'Northeast': {'base_income': 75000, 'approval_modifier': 1.1},
            'Southeast': {'base_income': 55000, 'approval_modifier': 0.9},
            'Midwest': {'base_income': 60000, 'approval_modifier': 1.0},
            'Southwest': {'base_income': 58000, 'approval_modifier': 0.95},
            'West': {'base_income': 70000, 'approval_modifier': 1.05}
        }
        
        # Loan types
        self.loan_types = ['Personal', 'Auto', 'Mortgage', 'Student', 'Business']
        
    def _generate_demographics(self) -> pd.DataFrame:
        """Generate demographic attributes."""
        demographics = pd.DataFrame()
        
        # Race
        demographics['race'] = np.random.choice(
            list(self.demographics['race'].keys()),
            self.n_samples,
            p=list(self.demographics['race'].values())
        )
        
        # Gender
        demographics['gender'] = np.random.choice(
            list(self.demographics['gender'].keys()),
            self.n_samples,
            p=list(self.demographics['gender'].values())
        )
        
        # Age
        age_groups = np.random.choice(
            list(self.demographics['age_groups'].keys()),
            self.n_samples,
            p=list(self.demographics['age_groups'].values())
        )
        
        # Convert age groups to specific ages
        demographics['age'] = self._age_from_group(age_groups)
        demographics['age_group'] = age_groups
        
        return demographics
    
    def _age_from_group(self, age_groups: np.ndarray) -> np.ndarray:
        """Convert age groups to specific ages."""
        ages = []
        for group in age_groups:
            if group == '18-24':
                ages.append(np.random.randint(18, 25))
            elif group == '25-34':
                ages.append(np.random.randint(25, 35))
            elif group == '35-44':
                ages.append(np.random.randint(35, 45))
            elif group == '45-54':
                ages.append(np.random.randint(45, 55))
            elif group == '55-64':
                ages.append(np.random.randint(55, 65))
            else:  # 65+
                ages.append(np.random.randint(65, 80))
        return np.array(ages)
    
    def _generate_geographic(self) -> pd.DataFrame:
        """Generate geographic attributes."""
        geographic = pd.DataFrame()
        
        # Region
        geographic['region'] = np.random.choice(
            list(self.regions.keys()),
            self.n_samples,
            p=[0.2, 0.2, 0.2, 0.2, 0.2]
        )
        
        # Generate ZIP codes (simplified)
        geographic['zip_code'] = [
            self.faker.zipcode() for _ in range(self.n_samples)
        ]
        
        # Urban/Rural flag
        geographic['urban_rural'] = np.random.choice(
            ['Urban', 'Suburban', 'Rural'],
            self.n_samples,
            p=[0.3, 0.5, 0.2]
        )
        
        return geographic
    
    def _generate_financial(self, demographics: pd.DataFrame, 
                           geographic: pd.DataFrame) -> pd.DataFrame:
        """Generate financial attributes with realistic correlations."""
        financial = pd.DataFrame()
        
        # Base income influenced by age, region, and education
        base_incomes = []
        for idx in range(self.n_samples):
            region = geographic.iloc[idx]['region']
            age = demographics.iloc[idx]['age']
            
            # Regional base income
            regional_base = self.regions[region]['base_income']
            
            # Age modifier (income typically peaks in 45-54)
            age_modifier = 1.0
            if age < 25:
                age_modifier = 0.5
            elif age < 35:
                age_modifier = 0.8
            elif age < 45:
                age_modifier = 1.0
            elif age < 55:
                age_modifier = 1.2
            elif age < 65:
                age_modifier = 1.1
            else:
                age_modifier = 0.7
            
            # Generate income with some randomness
            income = regional_base * age_modifier * np.random.lognormal(0, 0.3)
            base_incomes.append(max(15000, min(500000, income)))
        
        financial['annual_income'] = base_incomes
        
        # Credit score (correlated with income)
        financial['credit_score'] = self._generate_credit_scores(
            financial['annual_income'].values
        )
        
        # Employment years (correlated with age)
        financial['employment_years'] = np.maximum(
            0,
            demographics['age'] - 18 - np.random.exponential(2, self.n_samples)
        ).astype(int)
        
        # Debt-to-income ratio
        financial['debt_to_income'] = np.random.beta(2, 5, self.n_samples) * 0.8
        
        # Number of existing loans
        financial['existing_loans'] = np.random.poisson(1.5, self.n_samples)
        
        # Previous defaults (inverse correlation with credit score)
        default_prob = 1 - (financial['credit_score'] - 300) / 550
        financial['previous_defaults'] = np.random.binomial(
            1, default_prob * 0.2
        )
        
        return financial
    
    def _generate_credit_scores(self, incomes: np.ndarray) -> np.ndarray:
        """Generate credit scores with income correlation."""
        # Normalize income to 0-1 range
        income_normalized = (incomes - incomes.min()) / (incomes.max() - incomes.min())
        
        # Base credit score influenced by income
        base_scores = 550 + income_normalized * 200
        
        # Add random variation
        noise = np.random.normal(0, 50, len(incomes))
        credit_scores = base_scores + noise
        
        # Clip to valid range
        return np.clip(credit_scores, 300, 850).astype(int)
    
    def _generate_loan_details(self, financial: pd.DataFrame) -> pd.DataFrame:
        """Generate loan application details."""
        loan = pd.DataFrame()
        
        # Loan type
        loan['loan_type'] = np.random.choice(
            self.loan_types,
            self.n_samples,
            p=[0.25, 0.20, 0.35, 0.10, 0.10]
        )
        
        # Loan amount (based on income and loan type)
        loan_amounts = []
        for idx in range(self.n_samples):
            income = financial.iloc[idx]['annual_income']
            loan_type = loan.iloc[idx]['loan_type']
            
            if loan_type == 'Personal':
                amount = np.random.uniform(1000, min(50000, income * 0.5))
            elif loan_type == 'Auto':
                amount = np.random.uniform(5000, min(75000, income * 0.8))
            elif loan_type == 'Mortgage':
                amount = np.random.uniform(50000, min(1000000, income * 5))
            elif loan_type == 'Student':
                amount = np.random.uniform(5000, 100000)
            else:  # Business
                amount = np.random.uniform(10000, min(500000, income * 3))
            
            loan_amounts.append(amount)
        
        loan['loan_amount'] = loan_amounts
        
        # Loan term (months)
        loan['loan_term_months'] = np.where(
            loan['loan_type'] == 'Mortgage',
            np.random.choice([180, 240, 360], self.n_samples),
            np.where(
                loan['loan_type'] == 'Auto',
                np.random.choice([36, 48, 60, 72], self.n_samples),
                np.random.choice([12, 24, 36, 48, 60], self.n_samples)
            )
        )
        
        # Interest rate requested
        loan['interest_rate_requested'] = np.random.uniform(3, 15, self.n_samples)
        
        # Application date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        loan['application_date'] = pd.to_datetime([
            self.faker.date_between(start_date=start_date, end_date=end_date)
            for _ in range(self.n_samples)
        ])
        
        return loan
    
    def _apply_bias_patterns(self, df: pd.DataFrame, 
                            bias_config: Dict[str, float]) -> pd.DataFrame:
        """
        Apply configurable bias patterns to the approval decisions.
        
        Args:
            df: The complete dataset
            bias_config: Configuration for bias patterns
        """
        # Base approval probability based on credit score and DTI
        base_approval_prob = (
            (df['credit_score'] - 300) / 550 * 0.7 +
            (1 - df['debt_to_income']) * 0.3
        )
        
        # Apply bias modifiers
        approval_prob = base_approval_prob.copy()
        
        if bias_config.get('gender_bias', False):
            # Reduce approval probability for women
            female_mask = df['gender'] == 'Female'
            approval_prob[female_mask] *= 0.85
            
        if bias_config.get('race_bias', False):
            # Reduce approval probability for minorities
            minority_mask = df['race'].isin(['Black', 'Hispanic'])
            approval_prob[minority_mask] *= 0.75
            
        if bias_config.get('age_bias', False):
            # Reduce approval probability for young and old
            young_mask = df['age'] < 25
            old_mask = df['age'] > 65
            approval_prob[young_mask | old_mask] *= 0.8
            
        if bias_config.get('geographic_bias', False):
            # Geographic redlining
            southeast_mask = df['region'] == 'Southeast'
            southwest_mask = df['region'] == 'Southwest'
            approval_prob[southeast_mask | southwest_mask] *= 0.85
            
        if bias_config.get('intersectional_bias', False):
            # Compound bias for intersectional groups
            intersectional_mask = (
                (df['gender'] == 'Female') & 
                (df['race'].isin(['Black', 'Hispanic']))
            )
            approval_prob[intersectional_mask] *= 0.9
        
        # Generate approval decisions
        df['approved'] = np.random.binomial(1, np.clip(approval_prob, 0, 1))
        
        # Generate approval amounts (for approved loans)
        df['approved_amount'] = np.where(
            df['approved'] == 1,
            df['loan_amount'] * np.random.uniform(0.8, 1.0, self.n_samples),
            0
        )
        
        # Generate interest rates (higher for biased groups)
        base_rate = 5 + (850 - df['credit_score']) / 100
        
        if bias_config.get('interest_rate_bias', False):
            minority_mask = df['race'].isin(['Black', 'Hispanic'])
            base_rate[minority_mask] += 0.5
            
            female_mask = df['gender'] == 'Female'
            base_rate[female_mask] += 0.3
        
        df['approved_interest_rate'] = np.where(
            df['approved'] == 1,
            base_rate + np.random.normal(0, 0.5, self.n_samples),
            0
        )
        
        return df
    
    def generate_dataset(self, bias_config: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Generate complete synthetic lending dataset.
        
        Args:
            bias_config: Dictionary specifying which biases to introduce
            
        Returns:
            Complete synthetic dataset
        """
        if bias_config is None:
            bias_config = {
                'gender_bias': True,
                'race_bias': True,
                'age_bias': True,
                'geographic_bias': True,
                'intersectional_bias': True,
                'interest_rate_bias': True
            }
        
        logger.info(f"Generating {self.n_samples} synthetic lending records...")
        
        # Generate all components
        demographics = self._generate_demographics()
        geographic = self._generate_geographic()
        financial = self._generate_financial(demographics, geographic)
        loan = self._generate_loan_details(financial)
        
        # Combine all dataframes
        df = pd.concat([demographics, geographic, financial, loan], axis=1)
        
        # Add unique identifier
        df['application_id'] = [f"APP{str(i).zfill(8)}" for i in range(self.n_samples)]
        
        # Apply bias patterns and generate approval decisions
        df = self._apply_bias_patterns(df, bias_config)
        
        # Add metadata
        df['dataset_version'] = '1.0.0'
        df['is_synthetic'] = True
        df['bias_config'] = str(bias_config)
        
        # Reorder columns
        column_order = [
            'application_id', 'application_date',
            'race', 'gender', 'age', 'age_group',
            'region', 'zip_code', 'urban_rural',
            'annual_income', 'credit_score', 'employment_years',
            'debt_to_income', 'existing_loans', 'previous_defaults',
            'loan_type', 'loan_amount', 'loan_term_months',
            'interest_rate_requested',
            'approved', 'approved_amount', 'approved_interest_rate',
            'dataset_version', 'is_synthetic', 'bias_config'
        ]
        
        df = df[column_order]
        
        logger.info(f"Successfully generated {len(df)} records")
        logger.info(f"Approval rate: {df['approved'].mean():.2%}")
        
        return df
    
    def generate_validation_report(self, df: pd.DataFrame) -> Dict:
        """Generate validation report for synthetic data."""
        report = {
            'total_records': len(df),
            'approval_rate': df['approved'].mean(),
            'demographics': {
                'race': df['race'].value_counts(normalize=True).to_dict(),
                'gender': df['gender'].value_counts(normalize=True).to_dict(),
                'age_groups': df['age_group'].value_counts(normalize=True).to_dict()
            },
            'financial_stats': {
                'income': {
                    'mean': df['annual_income'].mean(),
                    'median': df['annual_income'].median(),
                    'std': df['annual_income'].std()
                },
                'credit_score': {
                    'mean': df['credit_score'].mean(),
                    'median': df['credit_score'].median(),
                    'std': df['credit_score'].std()
                }
            },
            'loan_stats': {
                'types': df['loan_type'].value_counts(normalize=True).to_dict(),
                'amount': {
                    'mean': df['loan_amount'].mean(),
                    'median': df['loan_amount'].median()
                }
            },
            'bias_indicators': {
                'gender_approval_rates': df.groupby('gender')['approved'].mean().to_dict(),
                'race_approval_rates': df.groupby('race')['approved'].mean().to_dict(),
                'region_approval_rates': df.groupby('region')['approved'].mean().to_dict()
            }
        }
        
        return report


# Example usage and testing
if __name__ == "__main__":
    # Initialize generator
    generator = SyntheticLendingDataGenerator(n_samples=10000)
    
    # Generate dataset with bias
    biased_data = generator.generate_dataset(bias_config={
        'gender_bias': True,
        'race_bias': True,
        'age_bias': True,
        'geographic_bias': True,
        'intersectional_bias': True,
        'interest_rate_bias': True
    })
    
    # Generate validation report
    report = generator.generate_validation_report(biased_data)
    
    # Display summary
    print("\n=== Synthetic Data Generation Complete ===")
    print(f"Total Records: {report['total_records']}")
    print(f"Overall Approval Rate: {report['approval_rate']:.2%}")
    
    print("\n=== Approval Rates by Demographics ===")
    print("\nBy Gender:")
    for gender, rate in report['bias_indicators']['gender_approval_rates'].items():
        print(f"  {gender}: {rate:.2%}")
    
    print("\nBy Race:")
    for race, rate in report['bias_indicators']['race_approval_rates'].items():
        print(f"  {race}: {rate:.2%}")
    
    print("\nBy Region:")
    for region, rate in report['bias_indicators']['region_approval_rates'].items():
        print(f"  {region}: {rate:.2%}")
    
    # Save to CSV
    biased_data.to_csv('data/synthetic/lending_data_biased.csv', index=False)
    print("\n✅ Data saved to 'data/synthetic/lending_data_biased.csv'")
    
    # Generate unbiased dataset for comparison
    unbiased_data = generator.generate_dataset(bias_config={
        'gender_bias': False,
        'race_bias': False,
        'age_bias': False,
        'geographic_bias': False,
        'intersectional_bias': False,
        'interest_rate_bias': False
    })
    
    unbiased_data.to_csv('data/synthetic/lending_data_unbiased.csv', index=False)
    print("✅ Unbiased data saved to 'data/synthetic/lending_data_unbiased.csv'")