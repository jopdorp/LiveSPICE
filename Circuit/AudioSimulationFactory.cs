using ComputerAlgebra;
using System;
using System.Linq;

namespace Circuit
{
    public static class AudioSimulationFactory
    {
        public static TransientSolution Solve(Analysis analysis, double sampleRate, int oversample)
        {
            if (analysis == null)
                throw new ArgumentNullException(nameof(analysis));
            if (sampleRate <= 0)
                throw new ArgumentOutOfRangeException(nameof(sampleRate));
            if (oversample <= 0)
                throw new ArgumentOutOfRangeException(nameof(oversample));

            return TransientSolution.Solve(analysis, (Real)1 / (sampleRate * oversample));
        }

        public static Simulation Create(Circuit circuit, Analysis analysis, double sampleRate, int oversample, int iterations)
        {
            if (circuit == null)
                throw new ArgumentNullException(nameof(circuit));
            TransientSolution solution = Solve(analysis, sampleRate, oversample);
            if (oversample <= 0)
                throw new ArgumentOutOfRangeException(nameof(oversample));
            if (iterations <= 0)
                throw new ArgumentOutOfRangeException(nameof(iterations));

            Expression inputExpression = circuit.Components.OfType<Input>().Select(i => i.In).SingleOrDefault();
            if (inputExpression == null)
                throw new NotSupportedException("Circuit has no inputs.");

            Expression outputExpression = 0;
            foreach (Speaker speaker in circuit.Components.OfType<Speaker>())
                outputExpression += speaker.Out;

            if (outputExpression.EqualsZero())
                throw new NotSupportedException("Circuit has no speaker outputs.");

            return new Simulation(solution)
            {
                Oversample = oversample,
                Iterations = iterations,
                Input = new[] { inputExpression },
                Output = new[] { outputExpression },
                Arguments = analysis.Parameters.Select(i => i.Expression)
            };
        }
    }
}