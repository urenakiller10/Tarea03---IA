import os
import fnmatch

# Configuración
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = "project_context.txt"

# Archivos y carpetas a ignorar
IGNORE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".Python",
    "chroma_ragA",
    "chroma_ragB",
    "*.db",
    "*.sqlite3",
    ".env",
    ".env.*",
    "*.key",
    "env",
    "venv",
    ".venv",
    "ENV",
    ".git",
    ".gitignore",
    ".vscode",
    ".idea",
    "*.log",
    ".DS_Store",
    "Thumbs.db",
    "data",  # Ignorar PDFs
    "*.pdf",
    OUTPUT_FILE,
    "generate_context.py",  # No incluir este script
]

# Extensiones de archivos a incluir
INCLUDE_EXTENSIONS = [
    ".py",
    ".txt",
    ".md",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
]

def should_ignore(path, item):
    """Verifica si un archivo o carpeta debe ser ignorado."""
    full_path = os.path.join(path, item)
    
    for pattern in IGNORE_PATTERNS:
        if fnmatch.fnmatch(item, pattern):
            return True
        if fnmatch.fnmatch(full_path, pattern):
            return True
    
    return False

def get_project_structure(root_dir, prefix="", ignore_first=True):
    """Genera la estructura de carpetas del proyecto."""
    structure = []
    
    try:
        items = sorted(os.listdir(root_dir))
    except PermissionError:
        return structure
    
    # Filtrar items ignorados
    items = [item for item in items if not should_ignore(root_dir, item)]
    
    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        item_path = os.path.join(root_dir, item)
        
        if os.path.isdir(item_path):
            connector = "└── " if is_last else "├── "
            if not ignore_first:
                structure.append(f"{prefix}{connector}{item}/")
            
            extension = "    " if is_last else "│   "
            new_prefix = prefix + extension if not ignore_first else ""
            structure.extend(get_project_structure(item_path, new_prefix, False))
        else:
            connector = "└── " if is_last else "├── "
            if not ignore_first:
                structure.append(f"{prefix}{connector}{item}")
    
    return structure

def get_file_content(file_path):
    """Lee el contenido de un archivo."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"[Error al leer archivo: {e}]"

def should_include_file(file_path):
    """Verifica si un archivo debe ser incluido."""
    ext = os.path.splitext(file_path)[1]
    return ext in INCLUDE_EXTENSIONS

def generate_context():
    """Genera el archivo de contexto completo."""
    output_lines = []
    
    # Header
    output_lines.append("=" * 80)
    output_lines.append("CONTEXTO DEL PROYECTO: Sistema RAG con Agente Conversacional")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # Estructura del proyecto
    output_lines.append("### ESTRUCTURA DEL PROYECTO ###")
    output_lines.append("")
    output_lines.append("Tarea03---IA/")
    structure = get_project_structure(PROJECT_ROOT)
    output_lines.extend(structure)
    output_lines.append("")
    output_lines.append("")
    
    # Separador
    output_lines.append("=" * 80)
    output_lines.append("### CÓDIGO DE LOS ARCHIVOS ###")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # Recorrer archivos
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Filtrar directorios ignorados
        dirs[:] = [d for d in dirs if not should_ignore(root, d)]
        
        for file in sorted(files):
            file_path = os.path.join(root, file)
            
            # Ignorar archivos
            if should_ignore(root, file):
                continue
            
            # Solo incluir archivos con extensiones específicas
            if not should_include_file(file_path):
                continue
            
            # Obtener ruta relativa
            rel_path = os.path.relpath(file_path, PROJECT_ROOT)
            
            output_lines.append("-" * 80)
            output_lines.append(f"ARCHIVO: {rel_path}")
            output_lines.append("-" * 80)
            output_lines.append("")
            
            content = get_file_content(file_path)
            output_lines.append(content)
            output_lines.append("")
            output_lines.append("")
    
    # Escribir archivo
    output_path = os.path.join(PROJECT_ROOT, OUTPUT_FILE)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))
    
    return output_path

if __name__ == "__main__":
    print("Generando contexto del proyecto...")
    print(f"Directorio raíz: {PROJECT_ROOT}")
    print()
    
    output = generate_context()
    
    file_size = os.path.getsize(output) / 1024  # KB
    
    print("=" * 60)
    print("✅ Contexto generado exitosamente!")
    print("=" * 60)
    print(f"📁 Archivo: {output}")
    print(f"📊 Tamaño: {file_size:.2f} KB")
    print()
    print("Puedes copiar el contenido de 'project_context.txt' y")
    print("pegarlo en un chat de IA para obtener contexto completo.")
    print("=" * 60)