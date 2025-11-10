import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_metrics(json_path="metrics.json"):
    """Carga metricas desde JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_metrics(metrics_data):
    """Genera analisis y visualizaciones."""
    df = pd.DataFrame(metrics_data)
    
    print("=" * 60)
    print("RESUMEN DE METRICAS - AGENTE B")
    print("=" * 60)
    
    print(f"\nTotal de preguntas: {len(df)}")
    
    print("\n--- TIEMPOS (ms) ---")
    print(df[['t_retrieval_ms', 't_generation_ms', 't_total_ms']].describe())
    
    print("\n--- TOKENS ---")
    print(df[['tokens_in', 'tokens_out']].describe())
    
    print("\n--- CALIDAD ---")
    print(df[['fidelity_binary', 'citations_correct_ratio', 'em_binary']].describe())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    df['t_retrieval_ms'].hist(bins=20, ax=axes[0, 0])
    axes[0, 0].set_title('Distribucion Tiempo de Recuperacion')
    axes[0, 0].set_xlabel('ms')
    
    df['t_generation_ms'].hist(bins=20, ax=axes[0, 1])
    axes[0, 1].set_title('Distribucion Tiempo de Generacion')
    axes[0, 1].set_xlabel('ms')
    
    df['fidelity_binary'].value_counts().plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Fidelidad (Citas Correctas)')
    axes[1, 0].set_ylabel('Cantidad')
    
    df.plot(x='question_id', y=['tokens_in', 'tokens_out'], ax=axes[1, 1])
    axes[1, 1].set_title('Tokens por Pregunta')
    axes[1, 1].set_ylabel('Tokens')
    
    plt.tight_layout()
    plt.savefig('metrics_analysis_B.png', dpi=300)
    print("\nGrafica guardada en 'metrics_analysis_B.png'")
    
    summary = df[['t_retrieval_ms', 't_generation_ms', 't_total_ms', 
                   'fidelity_binary', 'citations_correct_ratio', 'em_binary',
                   'tokens_in', 'tokens_out']].describe()
    
    print("\n--- TABLA RESUMEN ---")
    print(summary)
    
    summary.to_csv('summary_B.csv')
    print("\nTabla guardada en 'summary_B.csv'")

if __name__ == "__main__":
    metrics = load_metrics()
    analyze_metrics(metrics)