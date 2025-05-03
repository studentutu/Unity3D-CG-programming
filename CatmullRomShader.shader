Shader "CatmullRom Surface"
{
	Subshader
	{
		Pass
		{
			Cull Off
			CGPROGRAM
			#pragma vertex VSMain
			#pragma fragment PSMain
			#pragma target 5.0

			uint _TessellationFactor;

			static float3 ControlPoints[4][4] = 
			{
				{ float3(0.0,  0.0,  0.0), float3(10.0, 10.0,  0.0), float3(20.0, 10.0,  0.0), float3(30.0,  0.0,  0.0)},
				{ float3(0.0, 10.0, 10.0), float3(10.0, 20.0, 10.0), float3(20.0, 20.0, 10.0), float3(30.0, 10.0, 10.0)},
				{ float3(0.0, 10.0, 20.0), float3(10.0, 20.0, 20.0), float3(20.0, 20.0, 20.0), float3(30.0, 10.0, 20.0)},
				{ float3(0.0,  0.0, 30.0), float3(10.0, 10.0, 30.0), float3(20.0, 10.0, 30.0), float3(30.0,  0.0, 30.0)}
			};

			// https://learn.microsoft.com/en-us/windows/win32/direct3d9/d3dxvec3catmullrom
			float3 CatmullRomCurve(float3 p1, float3 p2, float3 p3, float3 p4, float s)
			{
				float3 a = (-1.0 * s * s * s + 2.0 * s * s - s) * p1;
				float3 b = (3.0 * s * s * s - 5.0 * s * s + 2.0) * p2;
				float3 c = (-3.0 * s * s * s + 4.0 * s * s + s) * p3;
				float3 d = (s * s * s - s * s) * p4;
				return 0.5 * (a + b + c + d);
			}

			float3 CatmullRomSurface(float3 cp[4][4], float u, float v)
			{
				float3 a = CatmullRomCurve (cp[0][0], cp[0][1], cp[0][2], cp[0][3], u);
				float3 b = CatmullRomCurve (cp[1][0], cp[1][1], cp[1][2], cp[1][3], u);
				float3 c = CatmullRomCurve (cp[2][0], cp[2][1], cp[2][2], cp[2][3], u);
				float3 d = CatmullRomCurve (cp[3][0], cp[3][1], cp[3][2], cp[3][3], u);
				return CatmullRomCurve(a, b, c, d, v);
			}

			float3 CatmullRomTangent(float3 p1, float3 p2, float3 p3, float3 p4, float s)
			{
				float3 a = (-3.0 * s * s + 4.0 * s - 1.0) * p1;
				float3 b = (9.0 * s * s - 10.0 * s) * p2;
				float3 c = (-9.0 * s * s + 8.0 * s + 1.0) * p3;
				float3 d = (3.0 * s * s - 2.0 * s) * p4;
				return 0.5 * (a + b + c + d);
			}

			float3 CatmullRomSurfaceNormal(float3 cp[4][4], float u, float v)
			{
				float3 row0 = CatmullRomCurve(cp[0][0], cp[0][1], cp[0][2], cp[0][3], u);
				float3 row1 = CatmullRomCurve(cp[1][0], cp[1][1], cp[1][2], cp[1][3], u);
				float3 row2 = CatmullRomCurve(cp[2][0], cp[2][1], cp[2][2], cp[2][3], u);
				float3 row3 = CatmullRomCurve(cp[3][0], cp[3][1], cp[3][2], cp[3][3], u);
				float3 dv = CatmullRomTangent(row0, row1, row2, row3, v);
				float3 col0 = CatmullRomCurve(cp[0][0], cp[1][0], cp[2][0], cp[3][0], v);
				float3 col1 = CatmullRomCurve(cp[0][1], cp[1][1], cp[2][1], cp[3][1], v);
				float3 col2 = CatmullRomCurve(cp[0][2], cp[1][2], cp[2][2], cp[3][2], v);
				float3 col3 = CatmullRomCurve(cp[0][3], cp[1][3], cp[2][3], cp[3][3], v);
				float3 du = CatmullRomTangent(col0, col1, col2, col3, u);
				return normalize(cross(du, dv));
			}

			// https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/mod.xhtml
			float Mod (float x, float y)
			{
				return x - y * floor(x / y);
			}

			// https://developer.nvidia.com/sites/all/modules/custom/gpugems/books/GPUGems/gpugems_ch25.html
			float Checkerboard(float2 uv)
			{
				float2 fw = max(abs(ddx_fine(uv)), abs(ddy_fine(uv)));
				float width = max(fw.x, fw.y);
				float2 p0 = uv - 0.5 * width, p1 = uv + 0.5 * width;
				#define BUMPINT(x) (floor((x)/2) + 2.f * max(((x)/2) - floor((x)/2) - .5f, 0.f))
				float2 i = (BUMPINT(p1) - BUMPINT(p0)) / width;
				return i.x * i.y + (1 - i.x) * (1 - i.y);
			}

			void Animation()
			{
				float amplitude = 10.0;
				for (int i = 0; i < 4; ++i) 
				{
					ControlPoints[0][i].y += sin(_Time.g + i) * amplitude;
					ControlPoints[3][i].y += sin(_Time.g + i + 1.0) * amplitude;
					ControlPoints[i][0].y += sin(_Time.g + i + 2.0) * amplitude;
					ControlPoints[i][3].y += sin(_Time.g + i + 3.0) * amplitude;
				}
			}

			// Generate surface of plane from grid of quads, then calculate final vertex position
			float4 VSMain (uint vertexId : SV_VertexID, out float3 normal : NORMAL, out float2 texcoord : TEXCOORD0) : SV_POSITION
			{
				Animation();
				int instance = int(floor(vertexId / 6.0));
				float x = sign(Mod(20.0, Mod(float(vertexId), 6.0) + 2.0));
				float y = sign(Mod(18.0, Mod(float(vertexId), 6.0) + 2.0));
				float u = (float(instance / _TessellationFactor) + x) / float(_TessellationFactor);
				float v = (Mod(float(instance), float(_TessellationFactor)) + y) / float(_TessellationFactor);
				float3 localPos = CatmullRomSurface (ControlPoints, u, v);
				normal = CatmullRomSurfaceNormal (ControlPoints, u, v);
				texcoord = float2(u, v);
				return UnityObjectToClipPos(float4(localPos, 1.0));
			}

			float4 PSMain (float4 vertex : SV_POSITION, float3 normal : NORMAL, float2 texcoord : TEXCOORD0) : SV_Target
			{
				float3 lightDir = normalize(_WorldSpaceLightPos0.xyz);
				float3 normalDir = normalize(normal);
				float diffuse = max(dot(lightDir, normalDir), 0.2);
				float pattern = Checkerboard(texcoord * _TessellationFactor);
				return float4(diffuse.xxx * (float3)(pattern), 1.0);
			}
			ENDCG
		}
	}
}