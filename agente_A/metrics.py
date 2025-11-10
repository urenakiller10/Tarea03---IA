import time
import re
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import tiktoken

@dataclass
class QuestionMetrics:
    """Metricas para una pregunta individual."""
    run_id: str
    timestamp: str
    agent_mode: str
    question_id: int
    question_text: str
    web_allowed: bool
    web_used: bool
    
    t_retrieval_ms: float
    t_generation_ms: float
    t_total_ms: float
    
    tokens_in: int
    tokens_out: int
    
    retrieved_docs: List[Dict[str, Any]]
    cited_docs: List[Dict[str, Any]]
    
    fidelity_binary: int
    citations_correct_ratio: float
    em_binary: int
    
    answer: str

class MetricsCollector:
    """Colector de metricas para evaluacion."""
    
    def __init__(self):
        self.metrics: List[QuestionMetrics] = []
        self.run_id = str(uuid.uuid4())[:8]
        
        self.gold_answers = {
            "distancia coseno": r"(coseno|cos|similitud.*coseno|\sum.*x.*y|producto.*escalar)",
            "distancia euclidiana": r"(euclidiana?|euclid|raiz.*cuadrada?|\sqrt|diferencia.*cuadrados?)",
            "regresion lineal": r"(y\s*=\s*[wm]|lineal|mx\s*\+\s*b|pendiente|intercepto)",
            "kernel": r"(funcion.*similitud|espacio.*caracteristicas|kernel|transformacion)",
            "backpropagation": r"(propagacion.*atras|gradiente|derivada|cadena|chain.*rule)",
            "gradiente descendente": r"(gradient.*descent|descenso.*gradiente|optimizacion|minimizar)",
        }
        
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Cuenta tokens usando tiktoken."""
        try:
            return len(self.tokenizer.encode(text))
        except:
            return len(text.split())
    
    def parse_citations(self, answer: str) -> List[Dict[str, Any]]:
        """
        Extrae citas del formato: [1] archivo.pdf, p.5
        """
        citations = []
        pattern = r'\[(\d+)\]\s*([^,]+),\s*p\.(\d+)'
        matches = re.finditer(pattern, answer, re.IGNORECASE)
        
        for match in matches:
            citations.append({
                "file": match.group(2).strip(),
                "page": int(match.group(3))
            })
        
        return citations
    
    def calculate_fidelity(self, cited: List[Dict], retrieved: List[Dict]) -> int:
        """
        Fidelidad = 1 si todas las citas estan en retrieved, 0 si no.
        """
        if not cited:
            return 0
        
        retrieved_set = {(doc["file"], doc["page"]) for doc in retrieved}
        
        for cite in cited:
            if (cite["file"], cite["page"]) not in retrieved_set:
                return 0
        
        return 1
    
    def calculate_citation_correctness(self, cited: List[Dict], retrieved: List[Dict]) -> float:
        """
        % de citas que coinciden con retrieved.
        """
        if not cited:
            return 0.0
        
        retrieved_set = {(doc["file"], doc["page"]) for doc in retrieved}
        correct = sum(1 for cite in cited if (cite["file"], cite["page"]) in retrieved_set)
        
        return correct / len(cited)
    
    def check_exact_match(self, question: str, answer: str) -> int:
        """
        Verifica si la respuesta contiene los conceptos clave para preguntas objetivas.
        """
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        from unidecode import unidecode
        answer_normalized = unidecode(answer_lower)
        
        for key, pattern in self.gold_answers.items():
            if key in question_lower:
                if re.search(pattern, answer_normalized, re.IGNORECASE):
                    return 1
                return 0
        
        return 1
    
    def add_metric(self, 
                   agent_mode: str,
                   question_id: int,
                   question_text: str,
                   web_allowed: bool,
                   web_used: bool,
                   t_retrieval_ms: float,
                   t_generation_ms: float,
                   tokens_in: int,
                   tokens_out: int,
                   retrieved_docs: List[Dict],
                   answer: str):
        """
        Agrega una metrica completa.
        """
        cited_docs = self.parse_citations(answer)
        fidelity = self.calculate_fidelity(cited_docs, retrieved_docs)
        citations_ratio = self.calculate_citation_correctness(cited_docs, retrieved_docs)
        em = self.check_exact_match(question_text, answer)
        
        metric = QuestionMetrics(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            agent_mode=agent_mode,
            question_id=question_id,
            question_text=question_text,
            web_allowed=web_allowed,
            web_used=web_used,
            t_retrieval_ms=t_retrieval_ms,
            t_generation_ms=t_generation_ms,
            t_total_ms=t_retrieval_ms + t_generation_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            retrieved_docs=retrieved_docs,
            cited_docs=cited_docs,
            fidelity_binary=fidelity,
            citations_correct_ratio=citations_ratio,
            em_binary=em,
            answer=answer
        )
        
        self.metrics.append(metric)
    
    def save_to_json(self, filepath: str = "metrics.json"):
        """Guarda metricas en JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([asdict(m) for m in self.metrics], f, indent=2, ensure_ascii=False)
    
    def save_to_csv(self, filepath: str = "metrics.csv"):
        """Guarda metricas en CSV."""
        import csv
        
        if not self.metrics:
            return
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'run_id', 'timestamp', 'agent_mode', 'question_id', 'question_text',
                'web_allowed', 'web_used', 't_retrieval_ms', 't_generation_ms', 't_total_ms',
                'tokens_in', 'tokens_out', 'retrieved_docs', 'cited_docs',
                'fidelity_binary', 'citations_correct_ratio', 'em_binary'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for m in self.metrics:
                row = asdict(m)
                row['retrieved_docs'] = json.dumps(row['retrieved_docs'])
                row['cited_docs'] = json.dumps(row['cited_docs'])
                row.pop('answer', None)
                writer.writerow(row)
    
    def get_summary(self) -> Dict[str, Any]:
        """Genera resumen de metricas."""
        if not self.metrics:
            return {}
        
        import statistics
        
        return {
            "total_questions": len(self.metrics),
            "avg_t_retrieval_ms": statistics.mean(m.t_retrieval_ms for m in self.metrics),
            "median_t_retrieval_ms": statistics.median(m.t_retrieval_ms for m in self.metrics),
            "avg_t_generation_ms": statistics.mean(m.t_generation_ms for m in self.metrics),
            "median_t_generation_ms": statistics.median(m.t_generation_ms for m in self.metrics),
            "avg_tokens_in": statistics.mean(m.tokens_in for m in self.metrics),
            "avg_tokens_out": statistics.mean(m.tokens_out for m in self.metrics),
            "fidelity_rate": statistics.mean(m.fidelity_binary for m in self.metrics),
            "avg_citation_correctness": statistics.mean(m.citations_correct_ratio for m in self.metrics),
            "exact_match_rate": statistics.mean(m.em_binary for m in self.metrics),
            "web_usage_rate": statistics.mean(m.web_used for m in self.metrics),
        }