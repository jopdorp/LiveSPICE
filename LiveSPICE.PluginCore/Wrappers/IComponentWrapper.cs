namespace LiveSPICE.PluginCore;

public interface IComponentWrapper
{
    string Name { get; }

    bool NeedRebuild { get; set; }
}
