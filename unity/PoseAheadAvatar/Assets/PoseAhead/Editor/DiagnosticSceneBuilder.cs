using System;
using System.Collections.Generic;
using System.IO;
using PoseAhead.Retargeting;
using PoseAhead.Transport;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace PoseAhead.Editor
{
    public static class DiagnosticSceneBuilder
    {
        // Our own primitive geometry: no asset downloads, accounts or external licenses.
        public static GameObject CreateHumanoid()
        {
            var root = new GameObject("DiagnosticHumanoid");
            root.SetActive(false);
            var mapping = new Dictionary<HumanBodyBones, Transform>();
            Transform Bone(HumanBodyBones id, Transform parent, Vector3 local)
            {
                var t = new GameObject(id.ToString()).transform;
                t.SetParent(parent, false); t.localPosition = local;
                mapping.Add(id, t); return t;
            }
            var hips = Bone(HumanBodyBones.Hips, root.transform, new Vector3(0, 1, 0));
            var spine = Bone(HumanBodyBones.Spine, hips, new Vector3(0, .18f, 0));
            var chest = Bone(HumanBodyBones.Chest, spine, new Vector3(0, .22f, 0));
            var neck = Bone(HumanBodyBones.Neck, chest, new Vector3(0, .16f, 0));
            var head = Bone(HumanBodyBones.Head, neck, new Vector3(0, .16f, 0));
            for (int side = 0; side < 2; side++)
            {
                bool left = side == 0; float sign = left ? 1 : -1;
                var shoulder = Bone(left ? HumanBodyBones.LeftShoulder : HumanBodyBones.RightShoulder,
                    chest, new Vector3(sign * .09f, .12f, 0));
                var arm = Bone(left ? HumanBodyBones.LeftUpperArm : HumanBodyBones.RightUpperArm,
                    shoulder, new Vector3(sign * .13f, 0, 0));
                var elbow = Bone(left ? HumanBodyBones.LeftLowerArm : HumanBodyBones.RightLowerArm,
                    arm, new Vector3(sign * .30f, 0, 0));
                Bone(left ? HumanBodyBones.LeftHand : HumanBodyBones.RightHand,
                    elbow, new Vector3(sign * .26f, 0, 0));
                var thigh = Bone(left ? HumanBodyBones.LeftUpperLeg : HumanBodyBones.RightUpperLeg,
                    hips, new Vector3(sign * .10f, 0, 0));
                var knee = Bone(left ? HumanBodyBones.LeftLowerLeg : HumanBodyBones.RightLowerLeg,
                    thigh, new Vector3(0, -.45f, 0));
                var foot = Bone(left ? HumanBodyBones.LeftFoot : HumanBodyBones.RightFoot,
                    knee, new Vector3(0, -.45f, 0));
                Bone(left ? HumanBodyBones.LeftToes : HumanBodyBones.RightToes,
                    foot, new Vector3(0, -.04f, -.16f));
            }
            var human = new List<HumanBone>();
            foreach (var pair in mapping)
                human.Add(new HumanBone { boneName = pair.Value.name,
                    humanName = HumanTrait.BoneName[(int)pair.Key], limit = new HumanLimit { useDefaultValues = true } });
            var skeleton = new List<SkeletonBone>();
            foreach (var t in root.GetComponentsInChildren<Transform>(true))
                skeleton.Add(new SkeletonBone { name = t.name, position = t.localPosition,
                    rotation = t.localRotation, scale = t.localScale });
            Avatar avatar = AvatarBuilder.BuildHumanAvatar(root, new HumanDescription {
                human = human.ToArray(), skeleton = skeleton.ToArray(), upperArmTwist = .5f,
                lowerArmTwist = .5f, upperLegTwist = .5f, lowerLegTwist = .5f,
                armStretch = .05f, legStretch = .05f, feetSpacing = 0, hasTranslationDoF = false });
            if (avatar == null || !avatar.isValid || !avatar.isHuman)
            {
                UnityEngine.Object.DestroyImmediate(avatar);
                UnityEngine.Object.DestroyImmediate(root);
                throw new InvalidOperationException("Diagnostic Humanoid construction failed.");
            }
            avatar.name = "PoseAheadDiagnosticAvatar";
            var animator = root.AddComponent<Animator>();
            animator.avatar = avatar; animator.applyRootMotion = false;
            animator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
            // Only add geometry after collecting the skeleton description.
            foreach (var pair in mapping)
            {
                Transform bone = pair.Value;
                if (pair.Key == HumanBodyBones.Head) continue;
                // Draw each skeleton edge at its parent's pivot; parent rotation
                // moves the segment, while the child's own rotation bends the next.
                if (bone.parent == root.transform) continue;
                Vector3 delta = bone.localPosition;
                var segment = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                segment.name = "Visual_" + pair.Key;
                segment.transform.SetParent(bone.parent, false);
                segment.transform.localPosition = delta * .5f;
                segment.transform.localRotation = Quaternion.FromToRotation(Vector3.up, delta);
                segment.transform.localScale = new Vector3(.07f, delta.magnitude * .5f, .07f);
                UnityEngine.Object.DestroyImmediate(segment.GetComponent<Collider>());
            }
            var skull = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            skull.name = "Visual_Head"; skull.transform.SetParent(head, false);
            skull.transform.localScale = new Vector3(.22f, .26f, .22f);
            UnityEngine.Object.DestroyImmediate(skull.GetComponent<Collider>());
            return root;
        }

        [MenuItem("PoseAhead/Create diagnostic Humanoid scene")]
        public static void CreateScene()
        {
            BuildScene();
        }

        public static Scene BuildScene()
        {
            const string directory = "Assets/PoseAhead/Scenes";
            Directory.CreateDirectory(Path.Combine(Application.dataPath, "PoseAhead/Scenes"));
            AssetDatabase.Refresh();
            // Unity refuses additive NewScene when an untitled scene is open.
            // Reuse only a pristine empty scene; never discard/save user scene contents.
            Scene previous = SceneManager.GetActiveScene();
            bool reuseEmpty = SceneManager.sceneCount == 1 && string.IsNullOrEmpty(previous.path) &&
                              !previous.isDirty && previous.GetRootGameObjects().Length == 0;
            if (!reuseEmpty)
                for (int i = 0; i < SceneManager.sceneCount; i++)
                    if (string.IsNullOrEmpty(SceneManager.GetSceneAt(i).path))
                        throw new InvalidOperationException("Save the current untitled scene before generating the diagnostic scene.");
            Scene scene = reuseEmpty ? previous :
                EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Additive);
            SceneManager.SetActiveScene(scene);
            try
            {
                var mannequin = CreateHumanoid(); // Build avatar while this is still a hierarchy root.
                var animator = mannequin.GetComponent<Animator>();
                string avatarPath = AssetDatabase.GenerateUniqueAssetPath(directory + "/DiagnosticAvatar.asset");
                AssetDatabase.CreateAsset(animator.avatar, avatarPath);
                var container = new GameObject("PoseAheadAvatarContainer");
                mannequin.transform.SetParent(container.transform, false);
                var receiver = container.AddComponent<PoseUdpReceiver>();
                var rootDriver = container.AddComponent<PoseRootDriver>();
                rootDriver.Configure(receiver, container.transform);
                var driver = mannequin.AddComponent<PoseHumanoidDriver>();
                driver.Configure(receiver, animator, container.transform);
                mannequin.SetActive(true);
                var telemetry = container.AddComponent<PoseDemoTelemetry>();
                telemetry.Configure(receiver, driver, rootDriver);
                var camera = new GameObject("Main Camera").AddComponent<Camera>();
                camera.tag = "MainCamera"; camera.transform.position = new Vector3(0, 1.3f, -3);
                camera.transform.rotation = Quaternion.LookRotation(new Vector3(0, 1.1f, 0) - camera.transform.position);
                camera.clearFlags = CameraClearFlags.SolidColor;
                camera.backgroundColor = new Color(.08f, .10f, .13f);
                var light = new GameObject("Directional Light").AddComponent<Light>();
                light.type = LightType.Directional; light.transform.rotation = Quaternion.Euler(45, -30, 0);
                var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
                floor.name = "Floor"; floor.transform.localScale = new Vector3(2, 1, 2);
                // Saved materials make limbs visible against the floor and distinguish sides.
                Material MaterialAsset(string name, Color color)
                {
                    var shader = Shader.Find("Standard");
                    if (shader == null) throw new InvalidOperationException("Built-in Standard shader is missing.");
                    var material = new Material(shader) { name = name, color = color };
                    AssetDatabase.CreateAsset(material,
                        AssetDatabase.GenerateUniqueAssetPath(directory + "/" + name + ".mat"));
                    return material;
                }
                var bodyMaterial = MaterialAsset("DiagnosticBody", new Color(.8f, .85f, .9f));
                var leftMaterial = MaterialAsset("DiagnosticLeft", new Color(.9f, .3f, .15f));
                var rightMaterial = MaterialAsset("DiagnosticRight", new Color(.12f, .6f, .85f));
                foreach (var renderer in mannequin.GetComponentsInChildren<Renderer>())
                    renderer.sharedMaterial = renderer.name.Contains("Left") ? leftMaterial :
                        renderer.name.Contains("Right") ? rightMaterial : bodyMaterial;
                floor.GetComponent<Renderer>().sharedMaterial =
                    MaterialAsset("DiagnosticFloor", new Color(.15f, .18f, .22f));
                string scenePath = AssetDatabase.GenerateUniqueAssetPath(directory + "/PoseAheadDiagnostic.unity");
                if (!EditorSceneManager.SaveScene(scene, scenePath))
                    throw new IOException("Could not save diagnostic scene.");
                AssetDatabase.SaveAssets();
                Selection.activeGameObject = mannequin;
                Debug.Log("PoseAhead diagnostic scene created: " + scenePath);
                return scene;
            }
            finally { if (previous.IsValid()) SceneManager.SetActiveScene(previous); }
        }
    }
}
