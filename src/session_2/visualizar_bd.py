import chromadb

def auditar_base_datos():
    try:
        # Conectarse a la ruta local
        cliente = chromadb.PersistentClient(path="./bd_vectores")
        coleccion = cliente.get_collection(name="rostros")
        
        # Obtener toda la metadata de la base de datos
        datos = coleccion.get()
        
        usuarios = set() # Usamos un set para no tener nombres repetidos
        if datos and 'metadatas' in datos:
            for meta in datos['metadatas']:
                if meta and 'usuario' in meta:
                    usuarios.add(meta['usuario'])
                    
        print("\n=== REPORTE DE LA BASE DE DATOS VECTORIAL ===")
        print(f"Total de vectores (embeddings) almacenados: {coleccion.count()}")
        print("Usuarios registrados:")
        for user in usuarios:
            print(f" - {user}")
        print("=============================================\n")
        
    except Exception as e:
        print(f"Error al leer la BD: {e}")

if __name__ == "__main__":
    auditar_base_datos()