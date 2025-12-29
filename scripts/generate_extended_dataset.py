#!/usr/bin/env python3
"""
Generate Extended Gastric Cancer Gastrectomy Dataset (500 patients)
Based on clinical statistics and realistic distributions

Usage: python generate_extended_dataset.py
Output: ../data/gastrectomy_patients_extended.json
"""

import json
import random
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def generate_patient(patient_id):
    """
    Generate realistic patient record based on clinical statistics:
    - Age: Mean 68-72, std 10 (web:66, web:71, web:74)
    - Complications: ~29-43% rate (web:66, web:68, web:73)
    - Stage distribution: IA/IB ~30%, II ~30%, III ~35%, IV ~5%
    - Laparoscopic: ~50-60% for early stages, ~30% for advanced
    """
    # Age distribution (mean 70, std 10)
    age = int(np.clip(np.random.normal(70, 10), 45, 90))
    
    # Sex distribution (M:F ratio ~2:1 in gastric cancer)
    sex = random.choices(['M', 'F'], weights=[65, 35])[0]
    
    # BMI distribution (mean 23-24 for Asian population, slightly higher for Western)
    bmi = round(np.clip(np.random.normal(24.0, 3.5), 17.0, 35.0), 1)
    
    # Tumor stage distribution
    stage = random.choices(
        ['IA', 'IB', 'IIA', 'IIB', 'IIIA', 'IIIB', 'IV'],
        weights=[15, 15, 15, 15, 18, 17, 5]
    )[0]
    
    # Tumor location
    location = random.choice(['antrum', 'body', 'cardia', 'fundus'])
    
    # Surgery type (laparoscopic more common for early stages)
    is_early_stage = stage in ['IA', 'IB', 'IIA']
    surgery_type = random.choices(
        ['laparoscopic', 'open'],
        weights=[60, 40] if is_early_stage else [30, 70]
    )[0]
    
    # Operation time (laparoscopic typically longer but less blood loss)
    if surgery_type == 'laparoscopic':
        op_time = int(np.random.normal(170, 25))
        blood_loss = int(np.clip(np.random.gamma(2, 40), 50, 300))
    else:
        op_time = int(np.random.normal(230, 30))
        blood_loss = int(np.clip(np.random.gamma(3, 80), 150, 600))
    
    # Neoadjuvant therapy (more common for advanced stages)
    neoadjuvant = stage in ['IIIA', 'IIIB', 'IV'] and random.random() < 0.75
    
    # Lymph nodes removed (D2 dissection standard: 25-45 nodes)
    lymph_nodes = int(np.clip(np.random.normal(32, 8), 15, 50))
    
    # Complications (higher for advanced stage, open surgery, elderly)
    complication_prob = 0.25  # Base rate
    if stage in ['IIIA', 'IIIB', 'IV']:
        complication_prob += 0.15
    if surgery_type == 'open':
        complication_prob += 0.08
    if age > 75:
        complication_prob += 0.10
    if bmi < 20 or bmi > 30:
        complication_prob += 0.05
    
    complications = random.random() < complication_prob
    
    # Hospital stay (longer if complications)
    base_stay = 8 if surgery_type == 'laparoscopic' else 11
    hospital_stay = base_stay + (random.randint(5, 12) if complications else random.randint(-2, 3))
    hospital_stay = max(5, hospital_stay)
    
    # Survival and status
    stage_survival_months = {
        'IA': (55, 65), 'IB': (50, 60),
        'IIA': (40, 55), 'IIB': (35, 50),
        'IIIA': (20, 40), 'IIIB': (15, 30),
        'IV': (8, 20)
    }
    
    min_surv, max_surv = stage_survival_months.get(stage, (12, 60))
    survival_months = random.randint(min_surv, max_surv)
    
    # Status based on stage and survival
    if stage in ['IV'] and survival_months < 18:
        status = 'deceased'
    elif stage in ['IIIB'] and survival_months < 20:
        status = 'deceased'
    elif survival_months < 24 and random.random() < 0.3:
        status = 'deceased'
    else:
        status = 'alive'
    
    return {
        'patient_id': f'EXT_P{patient_id:03d}',
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'tumor_stage': stage,
        'tumor_location': location,
        'surgery_type': surgery_type,
        'operation_time_min': op_time,
        'blood_loss_ml': blood_loss,
        'complications': complications,
        'hospital_stay_days': hospital_stay,
        'neoadjuvant_therapy': neoadjuvant,
        'lymph_nodes_removed': lymph_nodes,
        'survival_months': survival_months,
        'status': status
    }

def main():
    print('Generating 500 realistic gastric cancer patient records...')
    
    # Generate 500 patients
    patients = [generate_patient(i+1) for i in range(500)]
    
    # Save to JSON
    output_path = Path(__file__).parent.parent / 'data' / 'gastrectomy_patients_extended.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(patients, f, indent=2, ensure_ascii=False)
    
    print(f'✓ Dataset saved to {output_path}')
    
    # Print statistics
    comp_rate = sum(1 for p in patients if p['complications']) / len(patients) * 100
    avg_age = sum(p['age'] for p in patients) / len(patients)
    lap_rate = sum(1 for p in patients if p['surgery_type'] == 'laparoscopic') / len(patients) * 100
    deceased_rate = sum(1 for p in patients if p['status'] == 'deceased') / len(patients) * 100
    
    print(f'\nDataset Statistics:')
    print(f'  Total patients: {len(patients)}')
    print(f'  Average age: {avg_age:.1f} years')
    print(f'  Complication rate: {comp_rate:.1f}%')
    print(f'  Laparoscopic surgery: {lap_rate:.1f}%')
    print(f'  Deceased patients: {deceased_rate:.1f}%')

if __name__ == '__main__':
    main()
