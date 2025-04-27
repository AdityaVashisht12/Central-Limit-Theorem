# app.py
from flask import Flask, render_template, jsonify, request
import numpy as np
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# API route to generate data serverside (useful for more complex distributions)
@app.route('/api/generate-population', methods=['POST'])
def generate_population():
    data = request.json
    distribution_type = data.get('distribution', 'uniform')
    size = data.get('size', 10000)
    
    if distribution_type == 'uniform':
        values = np.random.uniform(0, 1, size=size).tolist()
    elif distribution_type == 'exponential':
        values = np.random.exponential(0.5, size=size).tolist()
    elif distribution_type == 'bimodal':
        # Generate bimodal distribution
        values1 = np.random.normal(0.3, 0.05, size=size//2)
        values2 = np.random.normal(0.8, 0.05, size=size//2)
        values = np.concatenate([values1, values2]).tolist()
    elif distribution_type == 'skewed':
        # Right skewed distribution
        values = np.random.power(2, size=size).tolist()
    else:
        values = np.random.uniform(0, 1, size=size).tolist()
    
    return jsonify({
        'data': values,
        'min': min(values),
        'max': max(values),
        'mean': np.mean(values),
        'std': np.std(values)
    })

# Optional route to compute sampling distribution serverside (for large simulations)
@app.route('/api/compute-sampling', methods=['POST'])
def compute_sampling():
    data = request.json
    population = np.array(data.get('population', []))
    sample_size = data.get('sampleSize', 10)
    num_samples = data.get('numSamples', 1000)
    
    if len(population) == 0:
        return jsonify({'error': 'No population data provided'})
    
    # Generate sample means
    sample_means = []
    for _ in range(num_samples):
        sample = np.random.choice(population, size=sample_size, replace=True)
        sample_means.append(float(np.mean(sample)))
    
    return jsonify({
        'sampleMeans': sample_means,
        'min': min(sample_means),
        'max': max(sample_means),
        'mean': np.mean(sample_means),
        'std': np.std(sample_means),
        'theoreticalStdError': np.std(population) / np.sqrt(sample_size)
    })

@app.route('/api/distributions')
def get_distributions():
    # Could be expanded with more distribution types
    distributions = {
        'uniform': 'Uniform Distribution',
        'exponential': 'Exponential Distribution',
        'bimodal': 'Bimodal Distribution',
        'skewed': 'Right-Skewed Distribution',
        'normal': 'Normal Distribution',
        'lognormal': 'Log-Normal Distribution',
        'poisson': 'Poisson Distribution (Discrete)'
    }
    return jsonify(distributions)

if __name__ == '__main__':
    app.run(debug=True)