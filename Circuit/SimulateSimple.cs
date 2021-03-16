using System;
using System.Collections;
using System.Collections.Generic;


namespace Circuit{
  public class SimulateSimple
    {
        public static void Main(string[] args) {
            Schematic schema = Schematic.Load(args[0]);
            Circuit circuit = schema.Build();
            ComputerAlgebra.Expression h = (ComputerAlgebra.Expression)1 / (48000 * 1);
            TransientSolution solution = TransientSolution.Solve(circuit.Analyze(), h);
            System.Console.WriteLine(solution);
            Simulation simulation = new Simulation(solution);
            List<double[]> input = new List<double[]>();
            List<double[]> output = new List<double[]>();
            double[] firstInput = {2.0};
            input.Add(firstInput);
            simulation.Run(1,input, output);
            System.Console.WriteLine(simulation.Output);
        }
    }

}