// In Unity, install Universal Render Pipeline (URP). Create new scene, add built-in Cube with position (0,0,0) and scale (2,2,2).
// Create new material, assign shader, assign material to cube. 
// This shader uses a technique called Sinusoidal Representation Network (SIREN).
// Shader has function which approximates a signed distance field using a neural network, in real time. SDF was generated from triangle mesh.
// You can read below how to train neural network with own meshes. 
Shader "Neural SDF URP"
{
	SubShader 
	{
		Tags
		{
			"RenderPipeline"="UniversalPipeline"
			"RenderType"="Transparent"
			"Queue"="Transparent"
		}

		HLSLINCLUDE
		#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
		CBUFFER_START(UnityPerMaterial)
		float4 _BaseMap_ST;
		float4 _BaseColor;
		float _Cutoff;
		float4 _Color;
		CBUFFER_END
		ENDHLSL

		Pass
		{
			Name "Unlit"
			Blend SrcAlpha OneMinusSrcAlpha
			Cull Off
			HLSLPROGRAM
			#pragma vertex VSMain
			#pragma fragment PSMain
			#pragma shader_feature _ALPHATEST_ON

			struct Attributes 
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
				float4 color : COLOR;
				UNITY_VERTEX_INPUT_INSTANCE_ID
			};

			struct Varyings
			{
				float4 vertex : SV_POSITION;
				float2 uv : TEXCOORD0;
				float4 color : COLOR;
				float3 worldPos : TEXCOORD1;
				UNITY_VERTEX_INPUT_INSTANCE_ID
				UNITY_VERTEX_OUTPUT_STEREO
			};

			float SignedDistanceField(float3 p) 
			{
				if (length(p) > 1.0) 
				{
					return length(p) - 0.85;
				}
				p.xzy = p.xyz;
				float4 f0_0 = sin(p.y * float4(-0.310,-3.139,-3.015, 1.747) + p.z * float4(-1.123, 1.368,-1.237, 1.241) + p.x * float4(-3.205, 0.310, 3.454, 1.311) + float4( 0.059, 1.676,-5.537, 1.169));
				float4 f0_1 = sin(p.y * float4( 0.109, 2.562,-3.938,-3.923) + p.z * float4(-3.841, 1.063, 1.127,-4.861) + p.x * float4( 2.224,-1.283,-2.949, 0.621) + float4(-7.890,-4.234,-6.298, 1.969));
				float4 f0_2 = sin(p.y * float4( 0.121, 0.642, 1.899,-0.470) + p.z * float4(-3.284, 3.157, 0.024,-0.720) + p.x * float4( 2.359, 4.376,-3.137, 2.492) + float4( 3.476,-2.348,-3.151,-2.517));
				float4 f0_3 = sin(p.y * float4( 2.910, 0.876, 1.065,-3.495) + p.z * float4( 3.050,-1.426, 0.910,-3.263) + p.x * float4(-3.941,-3.252,-0.910,-2.069) + float4( 4.434,-3.918,-1.586, 6.586));
				float4 f1_0 = sin(
					  mul(f0_0, float4x4( 0.165,-0.140,0.111,-0.624, 0.119, 0.322, 0.106,-0.609, 0.292, 0.114, 0.389,-0.116,-0.155,-0.062,-0.893, 0.079))
					+ mul(f0_1, float4x4(-0.921,-0.453,0.100, 0.960,-0.118, 0.196, 0.126,-0.259,-0.532,-0.093, 0.273,-0.245, 0.161, 0.338,-0.409,-0.102))
					+ mul(f0_2, float4x4(-0.550,-0.189,0.260,-0.155,-0.027, 0.170, 0.149, 0.218,-0.028,-0.250,-0.089, 0.081,-0.721,-0.403, 0.043,-0.001))
					+ mul(f0_3, float4x4(-0.239,-0.196,0.633, 0.284,-0.159,-0.657,-0.577,-0.231, 0.309, 0.784, 0.132, 0.752, 0.234,-0.333,-0.026,-0.139))
					+ float4(-3.768, 3.326,0.860,-3.404)) / 1.0 + f0_0;
				float4 f1_1 = sin(
					  mul(f0_0, float4x4( 0.347, 0.235,-0.081,0.086, 0.478, 0.146,-0.290,-0.187, 0.027,-0.171,-0.331,-0.131,0.437, 0.189, 0.300,-0.095))
					+ mul(f0_1, float4x4(-0.238, 0.098, 0.062,0.351, 0.268, 0.568,-0.570,-0.857,-0.104, 0.186,-0.470, 0.083,0.184,-0.022,-0.407,-0.092))
					+ mul(f0_2, float4x4( 0.314,-0.390,-0.076,0.862,-0.336,-0.162,-0.484,-0.329,-0.165,-0.155, 0.568, 0.385,0.031,-0.436, 0.844, 0.240))
					+ mul(f0_3, float4x4(-0.302, 0.146, 0.116,0.058, 0.093, 0.610, 0.531,-0.476, 0.628, 0.210,-0.930, 0.174,0.193,-0.532, 0.174,-0.211))
					+ float4(-0.494,-1.279,-0.574,1.837)) / 1.0 + f0_1;
				float4 f1_2 = sin(
					  mul(f0_0, float4x4(-0.713,-0.038, 0.557,-0.532,-0.096,-0.075,0.632,-0.198,-0.473,-0.558,0.441, 0.212, 0.002,-0.806,-0.645, 0.612))
					+ mul(f0_1, float4x4( 0.393,-0.125,-0.182, 0.666, 0.360, 0.997,0.255, 0.800,-0.217, 0.515,0.670,-0.384,-0.070, 0.134, 0.260,-0.084))
					+ mul(f0_2, float4x4(-0.654, 0.080, 0.307, 0.490,-0.057, 0.055,0.084, 0.322, 0.837, 0.674,0.367,-0.556, 0.241, 0.839, 0.456, 0.410))
					+ mul(f0_3, float4x4(-0.207,-0.199, 0.273,-0.384,-0.484, 0.053,0.134, 0.494, 0.558, 0.644,0.143, 0.366, 0.113,-0.386, 0.078, 0.668))
					+ float4( 0.993, 0.533,-0.191, 0.564)) / 1.0 + f0_2;
				float4 f1_3 = sin(
					  mul(f0_0, float4x4( 0.906,-0.468,-0.606, 0.059,-0.008, 0.431,-0.179, 1.060,-0.212,-0.033,-0.070,-0.856,-0.026,-0.103,-0.449, 0.130))
					+ mul(f0_1, float4x4(-0.218, 0.409, 0.780, 0.748, 0.008, 0.531,-0.039,-0.273,-0.073,-0.694,-0.336,-0.343,-0.558, 0.554, 0.003, 0.429))
					+ mul(f0_2, float4x4( 0.459,-0.113,-0.494, 0.102, 0.244,-0.340, 0.560, 0.173,-0.573,-0.101, 0.388,-0.479,-0.301, 0.023, 0.374,-0.660))
					+ mul(f0_3, float4x4(-0.197,-0.142, 0.349,-0.553,-0.274,-0.269,-0.584,-1.728,-0.197, 0.318,-0.266, 0.434,-0.174, 0.458,-0.100, 0.297))
					+ float4(-1.544, 0.676,-2.101, 0.169)) / 1.0 + f0_3;
				float4 f2_0 = sin(
					  mul(f1_0, float4x4( 0.503,-0.082, 0.733,-0.314, 0.340, 0.326, 0.476,-0.014, 0.564,-0.510, 0.339,0.218, 0.501, 0.072,-0.356, 1.210))
					+ mul(f1_1, float4x4(-0.361, 1.027,-0.460,-0.153,-0.024,-0.196, 0.721,-0.210, 0.061,-0.075,-0.136,0.249, 0.324,-0.127,-0.275, 0.417))
					+ mul(f1_2, float4x4( 0.606,-0.498, 0.221,-0.254, 0.279,-0.662,-0.070, 0.286,-1.310,-0.118,-0.271,0.092,-0.243, 0.028,-0.088,-0.320))
					+ mul(f1_3, float4x4(-0.033, 0.012,-0.721,-0.073, 0.309,-0.488, 0.429, 0.902, 0.032,-0.530,-0.083,0.095, 0.188,-0.312,-0.050, 0.313))
				 	+ float4(-0.315,-3.145, 2.215,-0.067)) / 1.4 + f1_0;
				float4 f2_1 = sin(
					  mul(f1_0, float4x4( 0.131,-0.399, 0.122,-0.514,-0.211, 0.197, 0.675,-0.222, 0.491,-1.296, 0.106, 0.506,-0.349, 0.081,1.009, 0.177))
					+ mul(f1_1, float4x4( 1.276, 0.055, 0.797, 0.855,-0.596,-0.432,-0.283, 0.527,-0.351,-0.347, 0.088, 0.174,-0.103,-0.057,0.661, 0.341))
					+ mul(f1_2, float4x4(-0.276,-0.444,-1.155,-0.410,-0.578,-0.251,-0.487, 0.281,-1.036, 0.602,-0.884,-0.312,-0.592, 0.803,0.310, 0.082))
					+ mul(f1_3, float4x4( 0.130, 0.491,-0.920,-0.308,-0.389,-1.274, 0.451,-0.000,-0.093, 0.110,-0.901,-0.394, 0.373, 0.410,0.046,-0.177))
					+ float4( 3.307, 0.156, 3.226,-3.042)) / 1.4 + f1_1;
				float4 f2_2 = sin(
					  mul(f1_0, float4x4(-0.646, 1.210,-0.605, 0.399, 0.312, 0.588,-0.956, 0.838,-0.098,-0.693,-0.115, 0.257, 0.774,-0.113,-0.419, 0.372))
					+ mul(f1_1, float4x4( 0.837, 0.679, 0.008, 0.247,-0.215,-0.293,-0.366,-0.029, 1.051,-1.084, 0.322, 0.272, 0.033,-0.279,-0.264, 0.321))
					+ mul(f1_2, float4x4( 0.811,-0.237, 0.226,-0.043, 0.922,-0.148, 0.273, 0.469,-0.118, 0.542, 0.872,-0.330,-0.197, 0.154, 0.758,-0.223))
					+ mul(f1_3, float4x4( 0.379,-0.180, 0.054,-0.120,-0.327,-0.793, 0.337,-0.074, 0.404,-0.435,-0.140, 1.008, 0.028, 0.283, 0.289,-0.525))
					+ float4( 3.485, 3.572,-2.964, 0.487)) / 1.4 + f1_2;
				float4 f2_3 = sin(
					  mul(f1_0, float4x4(-0.799,-0.022, 0.031,-0.318,0.018,-0.076,-0.127,-1.679,-0.291,-0.203,-0.100, 0.607, 0.342,-0.805,-0.063,-0.604))
					+ mul(f1_1, float4x4(-0.732,-1.039, 1.476, 1.442,0.312, 0.819, 0.082,-0.949, 0.519,-0.257,-0.379, 0.834, 0.080,-0.644,-0.149,-0.914))
					+ mul(f1_2, float4x4( 0.661, 0.788,-0.711, 0.348,0.407, 0.138,-0.341, 0.261, 0.432, 0.192, 0.109,-0.391, 0.277, 0.477,-0.242,-0.082))
					+ mul(f1_3, float4x4(-0.106, 0.398,-0.013, 0.147,0.346,-0.554,-0.594, 1.917,-0.359, 0.017, 0.331,-0.716,-0.206,-0.418,-0.286,-0.115))
					+ float4( 2.331,-1.299, 0.621,-2.623)) / 1.4 + f1_3;
				return
					  dot(f2_0, float4( 0.104,  0.069,  0.101, -0.058)) + dot(f2_1, float4(-0.072, -0.047,  0.023,  0.099))
					+ dot(f2_2, float4( 0.058, -0.035, -0.090, -0.061)) + dot(f2_3, float4(-0.103, -0.036, -0.073, -0.048)) - 0.061;
			}

			float4 RayMarching (float3 ro, float3 rd)
			{
				for (int i = 0; i < 128; i++)
				{
					float t = SignedDistanceField(ro);
					if (t<0.001) return (float4)(pow(1.0 - float(i) / float(128), 2.0)); 
					ro += t * rd;
				}
				return (float4) 0.0;
			}
	
			Varyings VSMain(Attributes IN) 
			{
				Varyings OUT;
				UNITY_SETUP_INSTANCE_ID(IN);
				UNITY_TRANSFER_INSTANCE_ID(IN, OUT);
				UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);
				OUT.vertex = TransformObjectToHClip(IN.vertex.xyz);
				OUT.uv = IN.uv;
				OUT.color = IN.color;
				OUT.worldPos = TransformObjectToWorld(IN.vertex.xyz);
				return OUT;
			}

			float4 PSMain(Varyings IN) : SV_Target
			{
				float3 worldPosition = IN.worldPos;
				float3 viewDirection = normalize(IN.worldPos - _WorldSpaceCameraPos);
				return RayMarching(worldPosition, viewDirection);
			}
			ENDHLSL
		}
	}
}

/*
Framework: https://www.shadertoy.com/view/wtVyWK by Blackle Morisanchetto
Data: "Spot" by Keenan Crane, https://github.com/alecjacobson/common-3d-test-models

Input mesh:
- Should have no hard surfaces;
- Should be placed in the same directory like neural_sdf.py (rename it to "spot.obj")
- Should fit sphere with origin (0,0,0) and radius 1.0

Install Python, then use command line:
- pip3 install numpy trimesh mesh-to-sdf
- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
- python neural_sdf.py

Python script will return neural network in form of text (GLSL language). Convert to Unity's HLSL manually.
*/