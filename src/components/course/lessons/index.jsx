// The lesson registry: one entry per CS156 lecture, with a title, a one-line note, and its component.
import Lesson01 from './Lesson01.jsx';
import Lesson02 from './Lesson02.jsx';
import Lesson03 from './Lesson03.jsx';
import Lesson04 from './Lesson04.jsx';
import Lesson05 from './Lesson05.jsx';
import Lesson06 from './Lesson06.jsx';
import Lesson07 from './Lesson07.jsx';
import Lesson08 from './Lesson08.jsx';
import Lesson09 from './Lesson09.jsx';
import Lesson10 from './Lesson10.jsx';
import Lesson11 from './Lesson11.jsx';
import Lesson12 from './Lesson12.jsx';
import Lesson13 from './Lesson13.jsx';
import Lesson14 from './Lesson14.jsx';
import Lesson15 from './Lesson15.jsx';
import Lesson16 from './Lesson16.jsx';
import Lesson17 from './Lesson17.jsx';
import Lesson18 from './Lesson18.jsx';

export const LESSONS = [
  { n: 1, title: 'the learning problem', blurb: 'label points, then watch a perceptron learn a boundary from them', C: Lesson01 },
  { n: 2, title: 'is learning feasible?', blurb: 'marbles in a bin and the hoeffding bound against the real deviation', C: Lesson02 },
  { n: 3, title: 'the linear model I', blurb: 'pla versus pocket on data no line can fully separate', C: Lesson03 },
  { n: 4, title: 'error and noise', blurb: 'move the target and the noise, watch E_in and E_out react', C: Lesson04 },
  { n: 5, title: 'training versus testing', blurb: 'count the dichotomies a hypothesis set can make on n points', C: Lesson05 },
  { n: 6, title: 'theory of generalization', blurb: 'the growth function against 2^N, and the break point', C: Lesson06 },
  { n: 7, title: 'the VC dimension', blurb: 'try to shatter points; the line breaks at d_vc + 1 = 4', C: Lesson07 },
  { n: 8, title: 'bias-variance tradeoff', blurb: 'resample tiny datasets and build the average hypothesis live', C: Lesson08 },
  { n: 9, title: 'the linear model II', blurb: 'a nonlinear feature transform bends a line into a circle', C: Lesson09 },
  { n: 10, title: 'neural networks', blurb: 'a short frame, then a hand-off into the inann tour in 3D', C: Lesson10 },
  { n: 11, title: 'overfitting', blurb: 'raise the degree and watch E_out blow up', C: Lesson11 },
  { n: 12, title: 'regularization', blurb: 'a weight-decay dial stiffens the overfit curve', C: Lesson12 },
  { n: 13, title: 'validation', blurb: 'k-fold splits and the held-out estimate of E_out', C: Lesson13 },
  { n: 14, title: 'support vector machines', blurb: 'drag points, see the widest-margin line and its support vectors', C: Lesson14 },
  { n: 15, title: 'kernel methods', blurb: 'swap linear, polynomial and rbf kernels on a ring', C: Lesson15 },
  { n: 16, title: 'radial basis functions', blurb: 'place gaussian centers and watch the fit form', C: Lesson16 },
  { n: 17, title: 'three learning principles', blurb: "occam's razor, sampling bias and data snooping", C: Lesson17 },
  { n: 18, title: 'epilogue', blurb: 'a map of the course and the way back into any lesson', C: Lesson18 },
];
