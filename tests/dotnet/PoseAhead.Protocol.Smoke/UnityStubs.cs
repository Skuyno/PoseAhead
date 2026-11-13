// Compile the actual MonoBehaviour adapter without an installed Unity editor.
// This checks C# and transport behavior, not Unity lifecycle/scene integration.
namespace UnityEngine
{
    public class MonoBehaviour
    { public Transform transform = new Transform(); public bool enabled; public GameObject gameObject = new GameObject(); }
    public class GameObject { public int scene; }
    public enum CameraType { Game, SceneView }
    public class Camera
    {
        public static event System.Action<Camera> onPostRender { add { } remove { } }
        public CameraType cameraType;
        public object targetTexture;
        public GameObject gameObject = new GameObject();
    }
    public static class Application
    {
        public static bool isBatchMode => true;
        public static bool isPlaying => true;
        public static string dataPath => ".";
        public static int targetFrameRate = -1;
    }
    public static class QualitySettings { public static int vSyncCount = 1; }
    public static class SystemInfo
    { public static Rendering.GraphicsDeviceType graphicsDeviceType => Rendering.GraphicsDeviceType.Null; }
    public static class JsonUtility
    { public static string ToJson(object value) => System.Text.Json.JsonSerializer.Serialize(value,
        new System.Text.Json.JsonSerializerOptions { IncludeFields = true }); }
    public class Transform
    {
        public Transform parent;
        public Vector3 localPosition;
        public Quaternion localRotation = new Quaternion(0, 0, 0, 1);
        public Vector3 lossyScale = new Vector3(1, 1, 1);
        public Quaternion rotation
        {
            get => parent == null ? localRotation : parent.rotation * localRotation;
            set => localRotation = parent == null ? value : Quaternion.Inverse(parent.rotation) * value;
        }
        public Vector3 position
        {
            get => parent == null ? localPosition : parent.position + parent.rotation * localPosition;
            set => localPosition = parent == null ? value :
                Quaternion.Inverse(parent.rotation) * (value - parent.position);
        }
    }
    public struct Vector3
    {
        public float x, y, z;
        public Vector3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
        public static Vector3 operator +(Vector3 a, Vector3 b) =>
            new Vector3(a.x + b.x, a.y + b.y, a.z + b.z);
        public static Vector3 operator -(Vector3 a, Vector3 b) =>
            new Vector3(a.x - b.x, a.y - b.y, a.z - b.z);
    }
    public struct Quaternion
    {
        public float x, y, z, w;
        public Quaternion(float x, float y, float z, float w)
        { this.x = x; this.y = y; this.z = z; this.w = w; }
        private System.Numerics.Quaternion Value => new System.Numerics.Quaternion(x, y, z, w);
        private static Quaternion From(System.Numerics.Quaternion q) =>
            new Quaternion(q.X, q.Y, q.Z, q.W);
        public static Quaternion Inverse(Quaternion q) => From(System.Numerics.Quaternion.Inverse(q.Value));
        public static Quaternion operator *(Quaternion a, Quaternion b) => From(a.Value * b.Value);
        public static Vector3 operator *(Quaternion q, Vector3 v)
        {
            var result = System.Numerics.Vector3.Transform(new System.Numerics.Vector3(v.x, v.y, v.z), q.Value);
            return new Vector3(result.X, result.Y, result.Z);
        }
    }
    public enum HumanBodyBones
    {
        Hips = 0, LeftUpperLeg = 1, RightUpperLeg = 2, LeftLowerLeg = 3, RightLowerLeg = 4,
        LeftFoot = 5, RightFoot = 6, Spine = 7, Chest = 8, Neck = 9, Head = 10,
        LeftUpperArm = 13, RightUpperArm = 14, LeftLowerArm = 15, RightLowerArm = 16,
        LeftHand = 17, RightHand = 18, LastBone = 55
    }
    public class Avatar { public bool isValid = true, isHuman = true; }
    public class Animator
    {
        public Avatar avatar = new Avatar();
        public object runtimeAnimatorController;
        public bool applyRootMotion;
        public readonly System.Collections.Generic.Dictionary<HumanBodyBones, Transform> Bones =
            new System.Collections.Generic.Dictionary<HumanBodyBones, Transform>();
        public Transform GetBoneTransform(HumanBodyBones id) => Bones.TryGetValue(id, out var t) ? t : null;
    }
    public static class Debug { public static void LogError(string error, object context) { } }
    public struct Rect { public Rect(float x, float y, float width, float height) { } }
    public static class GUI
    {
        public static void Box(Rect rect, string label) { }
        public static void Label(Rect rect, string label) { }
        public static bool Button(Rect rect, string label) => false;
    }
    public static class Time { public static float unscaledDeltaTime => .02f; public static int frameCount => 0; }
    public sealed class SerializeField : System.Attribute { }
}
namespace UnityEngine.Rendering { public enum GraphicsDeviceType { Null } }
