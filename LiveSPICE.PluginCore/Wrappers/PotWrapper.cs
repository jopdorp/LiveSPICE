using Circuit;

namespace LiveSPICE.PluginCore;

public class PotWrapper : ComponentWrapper<IPotControl>
{
    public PotWrapper(IPotControl pot, string name) : base(pot, name)
    {
    }

    public double PotValue
    {
        get => Sections[0].PotValue;
        set
        {
            foreach (IPotControl section in Sections)
                section.PotValue = value;
        }
    }
}
