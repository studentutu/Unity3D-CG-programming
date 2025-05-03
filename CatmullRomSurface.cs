using UnityEngine;
 
public class CatmullRomSurface : MonoBehaviour
{
	public Shader CatmullRomSurfaceShader;
	[Range(1, 1024)] public int TessellationFactor = 32;
	private Material _Material;

	void Start()
	{
		_Material = new Material(CatmullRomSurfaceShader);
	}

	void OnRenderObject()
	{
		_Material.SetInt("_TessellationFactor", TessellationFactor);
		_Material.SetPass(0);
		int vertexCount = TessellationFactor * TessellationFactor * 6;
		Graphics.DrawProceduralNow(MeshTopology.Triangles, vertexCount, 1);
	}
}